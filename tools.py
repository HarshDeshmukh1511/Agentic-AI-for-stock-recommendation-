"""
rag_tool.py
================================================================
Agentic RAG tool for NSE / SEBI company & stock reports.

Purpose
-------
Given a folder of NSE (National Stock Exchange) and SEBI (Securities
and Exchange Board of India) reports/filings about a company
(e.g. quarterly results, board meeting outcomes, corporate
announcements, shareholding patterns, SEBI orders/circulars,
annual reports), this module:

  1. Initializes a local, persistent ChromaDB vector store.
  2. Loads & chunks the source documents (PDF/TXT) with metadata
     that preserves company, source authority, doc type, and date.
  3. Embeds and stores the chunks in ChromaDB.
  4. Exposes a `search()` function (and a tool-schema wrapper) that
     an agentic AI can call to retrieve grounded context before
     making a "buy/hold/sell / next-quarter outlook" style call.

Design notes
------------
- Embeddings: sentence-transformers "all-MiniLM-L6-v2" by default
  (local, free, no API key). Swap `embedding_fn` for OpenAI/Voyage/
  Anthropic-compatible embeddings if you want higher quality.
- Chunking: a recursive character splitter tuned for financial
  filings (keeps tables/paragraphs mostly intact, adds overlap so
  numeric context - e.g. "Net Profit ... Rs. 120 Cr" - isn't split
  across chunks).
- Persistence: ChromaDB PersistentClient writes to disk, so the
  index survives process restarts (no re-embedding every run).
- Tool wrapper: `search_reports_tool` returns a plain dict, and
  `TOOL_SCHEMA` gives the JSON schema an agent framework (OpenAI
  function calling, Anthropic tool use, LangChain, etc.) needs to
  call it.

Install
-------
pip install chromadb sentence-transformers pypdf
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False


# ----------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------

@dataclass
class RAGConfig:
    persist_dir: str = "./chroma_db"                 # local vector DB folder
    collection_name: str = "nse_sebi_reports"
    embedding_model: str = "all-MiniLM-L6-v2"         # local, no API key needed
    chunk_size: int = 1200                            # chars per chunk
    chunk_overlap: int = 200                          # char overlap between chunks
    top_k_default: int = 5


# ----------------------------------------------------------------
# 2. VECTOR DB INITIALIZATION
# ----------------------------------------------------------------

class ReportVectorStore:
    """Wraps a local persistent ChromaDB collection for NSE/SEBI reports."""

    def __init__(self, config: RAGConfig = RAGConfig()):
        self.config = config
        Path(config.persist_dir).mkdir(parents=True, exist_ok=True)

        # Local, on-disk client -> data survives across runs.
        self.client = chromadb.PersistentClient(path=config.persist_dir)

        # Local embedding function (downloads model once, then runs offline).
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )

        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # -------------------------------------------------------------
    # 3. CHUNKING
    # -------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """
        Recursive-ish character splitter tuned for financial filings.

        Tries to split on paragraph breaks first, then sentences,
        then hard character windows -- always keeping `chunk_overlap`
        characters of context between consecutive chunks so a number
        (e.g. revenue figure) doesn't get orphaned from its label.
        """
        text = self._clean_text(text)
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        if len(text) <= size:
            return [text] if text else []

        # Prefer splitting on paragraph boundaries.
        paragraphs = re.split(r"\n\s*\n", text)

        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 1 <= size:
                current = f"{current}\n\n{para}".strip()
                continue

            if current:
                chunks.append(current)

            if len(para) <= size:
                current = para
            else:
                # Paragraph itself too long -> hard-window it with overlap.
                start = 0
                while start < len(para):
                    end = start + size
                    chunks.append(para[start:end])
                    start = end - overlap
                current = ""

        if current:
            chunks.append(current)

        # Stitch overlap between consecutive chunks for local context continuity.
        overlapped: List[str] = []
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
            else:
                prefix = chunks[i - 1][-overlap:] if overlap else ""
                overlapped.append((prefix + "\n" + c).strip())

        return overlapped

    # -------------------------------------------------------------
    # 4. DOCUMENT LOADING
    # -------------------------------------------------------------

    def _read_pdf(self, path: Path) -> str:
        if not _HAS_PYPDF:
            raise RuntimeError("pypdf not installed. Run: pip install pypdf")
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    def _read_txt(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def load_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf(path)
        if suffix in (".txt", ".md"):
            return self._read_txt(path)
        raise ValueError(f"Unsupported file type: {suffix} ({path})")

    # -------------------------------------------------------------
    # 5. INGEST (chunk + vectorize + store)
    # -------------------------------------------------------------

    def ingest_file(
        self,
        file_path: str,
        company: str,
        source: str,          # "NSE" or "SEBI"
        doc_type: str,        # e.g. "quarterly_results", "board_meeting",
                               # "shareholding_pattern", "sebi_order",
                               # "annual_report", "corporate_announcement"
        report_date: Optional[str] = None,   # "YYYY-MM-DD" if known
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Ingest a single file: load -> chunk -> embed -> upsert. Returns #chunks added."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)

        raw_text = self.load_file(path)
        chunks = self.chunk_text(raw_text)
        if not chunks:
            return 0

        ids, docs, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:10]
            chunk_id = f"{path.stem}_{i}_{chunk_hash}"

            meta = {
                "company": company,
                "source": source,               # NSE / SEBI
                "doc_type": doc_type,
                "report_date": report_date or "unknown",
                "file_name": path.name,
                "chunk_index": i,
                "ingested_at": datetime.utcnow().isoformat(),
            }
            if extra_metadata:
                meta.update(extra_metadata)

            ids.append(chunk_id)
            docs.append(chunk)
            metadatas.append(meta)

        self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
        return len(chunks)

    def ingest_directory(
        self,
        directory: str,
        company: str,
        source_map: Optional[Dict[str, str]] = None,
        doc_type_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        """
        Bulk-ingest every .pdf/.txt/.md file in `directory`.

        source_map / doc_type_map let you map filename substrings ->
        source/doc_type, e.g. {"sebi": "SEBI", "nse": "NSE"} and
        {"quarterly": "quarterly_results", "shareholding": "shareholding_pattern"}.
        Falls back to source="UNKNOWN", doc_type="general" if no match.
        """
        source_map = source_map or {}
        doc_type_map = doc_type_map or {}
        results: Dict[str, int] = {}

        for path in Path(directory).glob("**/*"):
            if path.suffix.lower() not in (".pdf", ".txt", ".md"):
                continue

            fname_lower = path.name.lower()
            source = next(
                (v for k, v in source_map.items() if k.lower() in fname_lower),
                "UNKNOWN",
            )
            doc_type = next(
                (v for k, v in doc_type_map.items() if k.lower() in fname_lower),
                "general",
            )

            n_chunks = self.ingest_file(
                file_path=str(path),
                company=company,
                source=source,
                doc_type=doc_type,
            )
            results[path.name] = n_chunks

        return results

    # -------------------------------------------------------------
    # 6. SEARCH
    # -------------------------------------------------------------

    def search(
        self,
        query: str,
        company: Optional[str] = None,
        source: Optional[str] = None,          # "NSE" or "SEBI"
        doc_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over ingested chunks, with optional metadata filters.
        Returns a list of {text, metadata, distance} sorted by relevance.
        """
        where: Dict[str, Any] = {}
        filters = []
        if company:
            filters.append({"company": company})
        if source:
            filters.append({"source": source})
        if doc_type:
            filters.append({"doc_type": doc_type})

        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        result = self.collection.query(
            query_texts=[query],
            n_results=top_k or self.config.top_k_default,
            where=where or None,
        )

        hits: List[Dict[str, Any]] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({
                "text": doc,
                "metadata": meta,
                "relevance_score": round(1 - dist, 4),  # cosine similarity approx
            })
        return hits

    def collection_stats(self) -> Dict[str, Any]:
        return {
            "collection": self.config.collection_name,
            "total_chunks": self.collection.count(),
            "persist_dir": self.config.persist_dir,
        }


# ----------------------------------------------------------------
# 7. AGENT-CALLABLE TOOL WRAPPER
# ----------------------------------------------------------------

# Instantiate once (module-level) so the agent framework can import
# `search_reports_tool` directly as a callable tool function.
_store = ReportVectorStore()


def search_reports_tool(
    query: str,
    company: Optional[str] = None,
    source: Optional[str] = None,
    doc_type: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Tool function for an agentic AI to retrieve grounded NSE/SEBI report
    excerpts before forming a next-quarter investment view.

    Args:
        query: Natural language question, e.g.
            "What did the company guide for margins next quarter?"
        company: Restrict to one company name (as ingested), optional.
        source: "NSE" or "SEBI", optional.
        doc_type: e.g. "quarterly_results", "sebi_order", optional.
        top_k: number of chunks to return.

    Returns:
        {
          "query": str,
          "results": [ {text, metadata, relevance_score}, ... ],
          "num_results": int
        }
    """
    hits = _store.search(
        query=query, company=company, source=source, doc_type=doc_type, top_k=top_k
    )
    return {"query": query, "results": hits, "num_results": len(hits)}


# JSON schema for tool-calling agent frameworks (OpenAI / Anthropic / LangChain).
TOOL_SCHEMA = {
    "name": "search_reports_tool",
    "description": (
        "Search NSE and SEBI reports/filings about a company (quarterly results, "
        "board meeting outcomes, shareholding patterns, SEBI orders/circulars, "
        "annual reports, corporate announcements) to ground next-quarter "
        "investment analysis in primary regulatory/exchange sources."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language question about the company/stock.",
            },
            "company": {
                "type": "string",
                "description": "Optional: restrict search to this company.",
            },
            "source": {
                "type": "string",
                "enum": ["NSE", "SEBI"],
                "description": "Optional: restrict to NSE or SEBI documents.",
            },
            "doc_type": {
                "type": "string",
                "description": (
                    "Optional: e.g. quarterly_results, board_meeting, "
                    "shareholding_pattern, sebi_order, annual_report, "
                    "corporate_announcement."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Number of chunks to retrieve (default 5).",
            },
        },
        "required": ["query"],
    },
}


# ----------------------------------------------------------------
# 8. EXAMPLE USAGE
# ----------------------------------------------------------------

if __name__ == "__main__":
    store = ReportVectorStore()

    # --- Ingest example (point this at your actual report folder) ---
    # summary = store.ingest_directory(
    #     directory="./reports/reliance",
    #     company="Reliance Industries",
    #     source_map={"sebi": "SEBI", "nse": "NSE"},
    #     doc_type_map={
    #         "quarterly": "quarterly_results",
    #         "shareholding": "shareholding_pattern",
    #         "board": "board_meeting",
    #         "order": "sebi_order",
    #         "annual": "annual_report",
    #     },
    # )
    # print("Ingested:", summary)

    print("Collection stats:", store.collection_stats())

    # --- Search example ---
    # results = store.search(
    #     query="What is management's guidance on revenue growth next quarter?",
    #     company="Reliance Industries",
    #     source="NSE",
    #     top_k=5,
    # )
    # for r in results:
    #     print(r["relevance_score"], r["metadata"], r["text"][:200])
import os
from typing import List, Dict, Any
from agno.agent import Agent
from agno.team import Team
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools import tool 
import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("GROQ_API_KEY")
MODEL_FAST = Groq(id="llama-3.1-8b-instant", api_key=api_key)         
MODEL_BALANCED = Groq(id="qwen/qwen3.6-27b", api_key=api_key)         
MODEL_HEAVY = Groq(id="llama-3.3-70b-versatile", api_key=api_key)
from tools import _store  # Import the _store object from tools.py
@tool
def search_nse_sebi_reports(
    query: str,
    company: str = None,
    source: str = None,       # "NSE" or "SEBI"
    doc_type: str = None,     # e.g. "quarterly_results", "sebi_order"
    top_k: int = 5,
) -> str:
    """
    Search NSE and SEBI reports/filings about a company (quarterly results,
    board meeting outcomes, shareholding patterns, SEBI orders/circulars,
    annual reports, corporate announcements). Use this to ground any
    next-quarter investment analysis in primary regulatory/exchange sources
    rather than general knowledge.

    Args:
        query: Natural language question, e.g. "What is management's
            guidance on margins next quarter?"
        company: Restrict results to this company name, if known.
        source: "NSE" or "SEBI", to restrict to one authority.
        doc_type: e.g. quarterly_results, board_meeting,
            shareholding_pattern, sebi_order, annual_report.
        top_k: number of passages to retrieve.

    Returns:
        A formatted string of the top matching report excerpts with
        their source metadata, for the agent to reason over.
    """
    hits = _store.search(
        query=query, company=company, source=source, doc_type=doc_type, top_k=top_k
    )
    if not hits:
        return "No matching passages found in the ingested NSE/SEBI reports."

    formatted = []
    for h in hits:
        m = h["metadata"]
        formatted.append(
            f"[{m['source']} | {m['doc_type']} | {m['company']} | "
            f"{m['report_date']} | score={h['relevance_score']}]\n{h['text']}"
        )
    return "\n\n---\n\n".join(formatted)
screener_agent = Agent(
    name = "Screener Agent",
    role = "Screens stock tickers based on fundamental metrics and technical analysis.",
    model = MODEL_FAST,
    tools = [YFinanceTools(enable_stock_price=True, enable_company_news=True, enable_income_statements=True, enable_key_financial_ratios=True, enable_stock_fundamentals=True, enable_analyst_recommendations=True)],
    instructions=[
        "Identify candidate stocks passing quarter-ahead investment criteria:",
        "1. Strong balance sheet (Low Debt-to-Equity, positive cash flow).",
        "2. Earnings growth momentum (Positive YoY EPS trends).",
        "3. Favorable analyst sentiment (Buy/Strong Buy ratings).",
        "Output a shortlisted pool of 8 to 10 ticker symbols with primary metrics in a markdown table."
    ],
    markdown=True,
)
news_agent = Agent(
    name="Market News Specialist",
    role="Gathers news context, sector momentum, and macro sentiment for candidates.",
    model=MODEL_FAST,    
    tools=[DuckDuckGoTools()],
    instructions=[
        "Search recent financial news, sector developments, and macroeconomic catalysts.",
        "Focus on news published within the past 60 days.",
        "Ground every news claim with structured citation URLs and timestamps."
    ],
    markdown=True,
)
selection_team = Team(
    name="Indian Stock Selection Panel",
    members=[screener_agent, news_agent],
    model=MODEL_FAST,
    role="Select 5 promising Indian stocks for investment next quarter.",
    instructions=[
        "Recommend 5 stocks listed on NSE/BSE.",
        "Ensure symbols follow the standard format (e.g., ICICIBANK.NS, BHARTIARTL.NS).",
        "List target prices (in INR ₹), P/E ratios, and specific growth triggers."
    ]
)
rag_agent = Agent(
    name="SEBI Filing RAG Analyst",
    model=MODEL_HEAVY,
    tools=[search_nse_sebi_reports,YFinanceTools(enable_stock_price=True, enable_company_news=True, enable_income_statements=True, enable_key_financial_ratios=True, enable_stock_fundamentals=True, enable_analyst_recommendations=True)],
    instructions=["Cross-reference SEBI filing text chunks and cite exact sections."],
)
verifier_agent = Agent(
    name="Compliance Verifier",
    model=MODEL_HEAVY,
    tools=[
        YFinanceTools(
            enable_stock_price=True,
            enable_income_statements=True,
            enable_key_financial_ratios=True,
            enable_stock_fundamentals=True,
            enable_analyst_recommendations=True
        ),
        DuckDuckGoTools(),
        search_nse_sebi_reports
    ],
    instructions=[
        "Verify every financial metric before approval.",
        "Cross-check all news claims with sources.",
        "Validate recommendations against NSE/SEBI filings.",
        "Flag unsupported statements.",
        "Rewrite incorrect figures using verified data.",
        "Produce a verification report with PASS/FAIL status for each claim."
    ],
    markdown=True
)

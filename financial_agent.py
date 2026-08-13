from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
import os
web_search_agent = Agent(
    name="Web Search Agent",
    role="A financial agent that can perform web searches and retrieve information from the web.",
    model=Groq(id="llama-3.3-70b-versatile", api_key=""),
    description="An agent that can perform web searches and retrieve information from the web.",
    tools=[DuckDuckGoTools()],
    instructions="You are a financial agent that can perform web searches and retrieve information from the web. Use the DuckDuckGoTools to perform web searches and retrieve information from the web. Use the Groq model to process and analyze the retrieved information. Always include citations for any information retrieved.",
    markdown=True,
)

financial_agent = Agent(
    name="Financial Agent",
    model=Groq(id="llama-3.3-70b-versatile", api_key=""),
    role="As a financial agent use YFinanceTools to retrieve financial data and perform analysis. Use information gathered by web_search_agent to provide insights and recommendations. Use the Groq model to process and analyze the retrieved information.",
    tools=[YFinanceTools(enable_stock_price=True, enable_company_news=True, enable_income_statements=True, enable_key_financial_ratios=True, enable_stock_fundamentals=True, enable_analyst_recommendations=True)],
    instructions=["Use tables and charts to present financial data and analysis. Provide insights and recommendations based on the retrieved information. Always include citations for any information retrieved."],
    markdown=True,
)
from agno.team import Team

multi_agent = Team(
    members=[web_search_agent, financial_agent],
    name="Multi-Agent Financial Team",
    model=Groq(id="llama-3.3-70b-versatile", api_key=""),
    role="Coordinate web research and financial analysis for company performance reviews.",
    instructions=[
        "Use web_search_agent to gather market context and citations.",
        "Use financial_agent to retrieve financial statements and ratios.",
        "If Yahoo Finance does not return data for a ticker, say so clearly and still provide a cautious, evidence-based summary using any other information you can retrieve.",
        "Clearly indicate when you are making assumptions or inferences based on incomplete data.",
        "Always include citations for any information retrieved.",
    ],
    markdown=True,
)



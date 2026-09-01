"""
ReAct Research Agent - Reasoning + Acting Pattern

Implements the ReAct paradigm where the agent:
1. Thinks about what to do (Reasoning)
2. Takes an action (Acting)
3. Observes the result
4. Repeats until task is complete

Uses LangGraph for the agent loop and Tavily for web search.

Usage:
    export GOOGLE_API_KEY="your-key"  # Get free at https://aistudio.google.com/app/apikey
    export TAVILY_API_KEY="your-key"  # Get free key at tavily.com
    python agent.py
"""

import os
import time
from typing import Annotated, TypedDict
from functools import wraps
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Custom Exceptions
# ============================================================================
class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class APIKeyError(AgentError):
    """Raised when API key is missing or invalid."""
    pass


class RateLimitError(AgentError):
    """Raised when API rate limit is exceeded."""
    pass


class SearchError(AgentError):
    """Raised when search tool fails."""
    pass


# ============================================================================
# Retry Decorator with Exponential Backoff
# ============================================================================
def retry_with_backoff(max_retries: int = 3, base_delay: float = 2.0):
    """
    Retry decorator with exponential backoff for handling rate limits.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()

                    # Check if it's a rate limit error
                    if "429" in str(e) or "resource_exhausted" in error_str or "quota" in error_str:
                        last_exception = RateLimitError(f"Rate limit exceeded: {e}")

                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)
                            print(f"\n[Rate limit hit. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})]")
                            time.sleep(delay)
                            continue

                    # Check for API key errors
                    elif "api_key" in error_str or "unauthorized" in error_str or "401" in str(e):
                        raise APIKeyError(f"Invalid or missing API key: {e}")

                    # Search-specific errors
                    elif "tavily" in error_str:
                        raise SearchError(f"Search failed: {e}")

                    # Other errors - don't retry
                    else:
                        raise AgentError(f"API error: {e}")

            # Max retries exceeded
            raise last_exception or AgentError("Max retries exceeded")

        return wrapper
    return decorator


# ============================================================================
# Environment Validation
# ============================================================================
def validate_environment():
    """Validate that required environment variables are set."""
    errors = []

    if not os.getenv("GOOGLE_API_KEY"):
        errors.append(
            "GOOGLE_API_KEY not found.\n"
            "  Get a free key at: https://aistudio.google.com/app/apikey\n"
            "  Then run: export GOOGLE_API_KEY='your-key'"
        )

    if not os.getenv("TAVILY_API_KEY"):
        errors.append(
            "TAVILY_API_KEY not found.\n"
            "  Get a free key at: https://tavily.com\n"
            "  Then run: export TAVILY_API_KEY='your-key'"
        )

    if errors:
        raise APIKeyError("\n\n".join(errors))


# ============================================================================
# Agent State
# ============================================================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================================
# Agent Creation
# ============================================================================
def create_agent():
    """Create the ReAct agent graph with error handling."""
    validate_environment()

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

    # Initialize tools
    try:
        search_tool = TavilySearchResults(max_results=3)
        tools = [search_tool]
    except Exception as e:
        raise SearchError(f"Failed to initialize search tool: {e}")

    llm_with_tools = llm.bind_tools(tools)

    # System prompt for ReAct behavior
    system_prompt = SystemMessage(
        content="""You are a research assistant that uses web search to find accurate information.

For each query:
1. THINK: Analyze what information is needed
2. ACT: Use the search tool to find information
3. OBSERVE: Review the search results
4. RESPOND: Provide a clear, sourced answer

Always cite your sources. If information seems outdated or conflicting,
search again with different terms."""
    )

    def agent_node(state: AgentState):
        """The agent reasoning node with error handling."""
        messages = [system_prompt] + state["messages"]

        @retry_with_backoff(max_retries=3, base_delay=2.0)
        def invoke():
            return llm_with_tools.invoke(messages)

        response = invoke()
        return {"messages": [response]}

    def should_continue(state: AgentState):
        """Determine if we should continue to tools or end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    # Set entry point
    graph.set_entry_point("agent")

    # Add edges
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ============================================================================
# Agent Runner
# ============================================================================
def extract_text_content(content) -> str:
    """Extract text from various response content formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ]
        return "\n".join(filter(None, text_parts))
    return str(content)


def run_agent(query: str) -> str:
    """
    Run the research agent with a query.

    Args:
        query: The research question

    Returns:
        The agent's response

    Raises:
        AgentError: If the agent fails to process the query
    """
    # Validate environment first
    try:
        agent = create_agent()
    except APIKeyError as e:
        print(f"\nConfiguration Error:\n{e}")
        return None
    except SearchError as e:
        print(f"\nSearch Tool Error: {e}")
        return None
    except Exception as e:
        print(f"\nFailed to create agent: {e}")
        return None

    print(f"Researching: {query}")
    print("=" * 50)

    try:
        # Stream the agent's execution
        final_answer = None

        for event in agent.stream({"messages": [HumanMessage(content=query)]}):
            for node_name, node_output in event.items():
                if node_name == "agent":
                    message = node_output["messages"][-1]
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tc in message.tool_calls:
                            search_query = tc.get("args", {}).get("query", "")
                            print(f"[Searching: {search_query}]")
                    elif message.content:
                        final_answer = extract_text_content(message.content)
                        print(f"\nAnswer:\n{final_answer}")
                elif node_name == "tools":
                    print("[Processing search results...]")

        print("=" * 50)
        return final_answer

    except RateLimitError as e:
        print(f"\nRate Limit Error: {e}")
        print("Please wait a minute and try again.")
        return None
    except SearchError as e:
        print(f"\nSearch Error: {e}")
        return None
    except AgentError as e:
        print(f"\nAgent Error: {e}")
        return None
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        return None


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    queries = [
        "What are the key features of Python 3.12?",
        "Compare React and Vue.js for building web apps in 2024",
    ]

    for query in queries:
        result = run_agent(query)
        if result is None:
            print("Failed to get response. Check errors above.")
        print("\n")

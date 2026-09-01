"""
Calculator Agent - Introduction to Tool Calling

Demonstrates how LLMs can invoke external functions (tools) to perform
actions they cannot do natively, like mathematical calculations.

Usage:
    export GOOGLE_API_KEY="your-key"  # Get free at https://aistudio.google.com/app/apikey
    python agent.py
"""

import os
import time
from functools import wraps
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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


class ToolExecutionError(AgentError):
    """Raised when a tool fails to execute."""
    pass


# ============================================================================
# Retry Decorator with Exponential Backoff
# ============================================================================
def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
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
                            print(f"[Rate limit hit. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})]")
                            time.sleep(delay)
                            continue

                    # Check for API key errors
                    elif "api_key" in error_str or "unauthorized" in error_str or "401" in str(e):
                        raise APIKeyError(f"Invalid or missing API key: {e}")

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
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise APIKeyError(
            "GOOGLE_API_KEY not found.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey\n"
            "Then run: export GOOGLE_API_KEY='your-key'"
        )

    return api_key


# ============================================================================
# Tools
# ============================================================================
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide first number by second. Returns error if dividing by zero."""
    if b == 0:
        raise ToolExecutionError("Cannot divide by zero")
    return a / b


@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base**exponent


# ============================================================================
# Agent
# ============================================================================
def create_agent():
    """Create an agent with calculator tools."""
    validate_environment()

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

    tools = [add, multiply, divide, power]
    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools, {t.name: t for t in tools}


@retry_with_backoff(max_retries=3, base_delay=2.0)
def invoke_llm(llm, messages):
    """Invoke LLM with retry logic."""
    return llm.invoke(messages)


def run_agent(query: str) -> str:
    """
    Run the calculator agent with a query.

    Args:
        query: The calculation query

    Returns:
        The agent's response

    Raises:
        AgentError: If the agent fails to process the query
    """
    try:
        llm, tools = create_agent()
    except APIKeyError as e:
        return f"Configuration Error: {e}"

    messages = [
        SystemMessage(
            content="You are a calculator assistant. Use the provided tools "
            "to perform calculations. Always show your work step by step."
        ),
        HumanMessage(content=query),
    ]

    try:
        # Get initial response
        response = invoke_llm(llm, messages)
        messages.append(response)

        # Process tool calls if any
        while response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"Calling: {tool_name}({tool_args})")

                # Execute the tool with error handling
                try:
                    result = tools[tool_name].invoke(tool_args)
                    print(f"Result: {result}")
                except ToolExecutionError as e:
                    result = f"Error: {e}"
                    print(f"Tool Error: {e}")
                except Exception as e:
                    result = f"Error executing {tool_name}: {e}"
                    print(f"Tool Error: {e}")

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )

            # Get next response
            response = invoke_llm(llm, messages)
            messages.append(response)

        # Extract text from response content
        content = response.content
        if isinstance(content, list):
            text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return "\n".join(text_parts) if text_parts else str(content)
        return content

    except RateLimitError as e:
        return f"Rate Limit Error: {e}\nPlease wait a minute and try again."
    except APIKeyError as e:
        return f"API Key Error: {e}"
    except AgentError as e:
        return f"Agent Error: {e}"
    except Exception as e:
        return f"Unexpected Error: {e}"


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    queries = [
        "What is 25 multiplied by 4?",
        "Calculate 100 divided by 5, then add 30",
        "What is 2 to the power of 10?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        result = run_agent(query)
        print(f"Answer: {result}")

# Calculator Tool Agent

A simple LLM agent demonstrating **tool calling** - the ability for language models to invoke external functions.

## What is Tool Calling?

Tool calling (also called function calling) allows an LLM to:
1. **Decide** when to use external tools based on user input
2. **Generate** structured arguments for the tool
3. **Process** tool results to formulate a response

```
┌─────────────────────────────────────────────────────────────┐
│                        TOOL CALLING FLOW                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   User: "What is 25 × 4?"                                   │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────┐                                       │
│   │      LLM        │  ← Analyzes query, decides tool needed│
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   Tool Call: multiply(a=25, b=4)   ← Generates structured   │
│            │                          function call          │
│            ▼                                                 │
│   ┌─────────────────┐                                       │
│   │  multiply()     │  ← External function executes         │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   Result: 100.0        ← Returns to LLM                     │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────┐                                       │
│   │      LLM        │  ← Formats final response             │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   "25 multiplied by 4 equals 100"                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Why Tool Calling Matters

| Without Tools | With Tools |
|---------------|------------|
| LLM can only generate text | LLM can take actions |
| May hallucinate calculations | Accurate computation |
| Static knowledge only | Access to live data |
| No external interactions | API calls, DB queries, etc. |

## How This Agent Works

### 1. Define Tools

Tools are Python functions decorated with `@tool`:

```python
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b
```

The decorator extracts:
- **Function name** → tool identifier
- **Parameters + types** → JSON schema for LLM
- **Docstring** → description helping LLM decide when to use it

### 2. Bind Tools to LLM

```python
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
llm_with_tools = llm.bind_tools([add, multiply, divide, power])
```

This tells the LLM what tools are available and their schemas.

### 3. Agent Loop

```python
while response.tool_calls:
    for tool_call in response.tool_calls:
        # Execute the tool
        result = tools[tool_call["name"]].invoke(tool_call["args"])

        # Send result back to LLM
        messages.append(ToolMessage(content=str(result), ...))

    # Get next response (may have more tool calls or final answer)
    response = llm.invoke(messages)
```

The loop continues until the LLM returns a response without tool calls.

### 4. Multi-Step Reasoning

For complex queries, the LLM chains multiple tools:

```
Query: "Calculate 100 ÷ 5, then add 30"

Step 1: LLM calls divide(100, 5) → 20
Step 2: LLM calls add(20, 30) → 50
Step 3: LLM responds "The answer is 50"
```

## Project Structure

```
calculator_tool_agent/
├── agent.py      # Main agent implementation
└── README.md     # This file
```

## Run

```bash
# Set API key (free at https://aistudio.google.com/app/apikey)
export GOOGLE_API_KEY="your-key"

# Install dependencies (from agents/ directory)
pip install -r ../requirements.txt

# Run
python agent.py
```

## Example Output

```
Query: What is 25 multiplied by 4?
----------------------------------------
Calling: multiply({'a': 25, 'b': 4})
Result: 100.0
Answer: 25 × 4 = 100

Query: Calculate 100 divided by 5, then add 30
----------------------------------------
Calling: divide({'a': 100, 'b': 5})
Result: 20.0
Calling: add({'a': 20, 'b': 30})
Result: 50.0
Answer: 100 ÷ 5 = 20, then 20 + 30 = 50
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| `@tool` | Decorator that converts a function into an LLM-callable tool |
| `bind_tools()` | Attaches tool definitions to the LLM |
| `tool_calls` | List of tools the LLM wants to invoke |
| `ToolMessage` | Message type for returning tool results to LLM |
| **Agent Loop** | Cycle of: LLM → Tool → Result → LLM until complete |

## Extending This Agent

Add new tools by defining functions:

```python
@tool
def square_root(n: float) -> float:
    """Calculate the square root of a number."""
    return n ** 0.5

# Add to tools list
tools = [add, multiply, divide, power, square_root]
```

## Tech Stack

- **LangChain** - Framework for LLM applications
- **Google Gemini** - LLM with tool calling support
- **Python** - Runtime

## References

- [LangChain Tool Calling](https://python.langchain.com/docs/concepts/tool_calling/)
- [Google Gemini Function Calling](https://ai.google.dev/gemini-api/docs/function-calling)

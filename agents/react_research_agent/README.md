# ReAct Research Agent

An advanced LLM agent implementing the **ReAct** (Reasoning + Acting) pattern using **LangGraph** for stateful orchestration.

## What is ReAct?

ReAct is a paradigm where the agent interleaves:
- **Reasoning** - thinking about what to do next
- **Acting** - executing tools/actions
- **Observing** - processing results

This creates a loop that continues until the task is complete.

```
┌─────────────────────────────────────────────────────────────┐
│                      ReAct LOOP                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   User: "What are Python 3.12 features?"                    │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────┐                                       │
│   │     THINK       │  "I need to search for Python 3.12"   │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────┐                                       │
│   │      ACT        │  search("Python 3.12 new features")   │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────┐                                       │
│   │    OBSERVE      │  [Results: improved errors, f-strings]│
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│      Need more info? ──────Yes────► Back to THINK           │
│            │                                                 │
│            No                                                │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────┐                                       │
│   │    RESPOND      │  Final answer with sources            │
│   └─────────────────┘                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Why ReAct?

| Simple Tool Calling | ReAct Pattern |
|---------------------|---------------|
| Single tool call | Multiple iterative calls |
| No reasoning trace | Explicit thinking steps |
| One-shot answer | Refined through observation |
| Limited context | Builds understanding |

## What is LangGraph?

LangGraph is a framework for building **stateful, multi-step** LLM applications as graphs.

```
┌─────────────────────────────────────────┐
│              LangGraph                   │
│                                          │
│   ┌─────────┐         ┌─────────┐       │
│   │  Agent  │────────►│  Tools  │       │
│   │  Node   │◄────────│  Node   │       │
│   └────┬────┘         └─────────┘       │
│        │                                 │
│        ▼ (no more tool calls)           │
│     [END]                               │
│                                          │
└─────────────────────────────────────────┘
```

### Key LangGraph Concepts

| Concept | Description |
|---------|-------------|
| **StateGraph** | Graph that maintains state across nodes |
| **Nodes** | Functions that process and update state |
| **Edges** | Connections between nodes (can be conditional) |
| **State** | Data passed between nodes (messages, context) |

## How This Agent Works

### 1. Define State

State holds the conversation history:

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

`add_messages` is a reducer that accumulates messages instead of replacing them.

### 2. Create Tools

We use Tavily for web search:

```python
search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]
```

### 3. Build the Graph

```python
# Create graph with state
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)      # LLM reasoning
graph.add_node("tools", ToolNode(tools)) # Tool execution

# Set entry point
graph.set_entry_point("agent")

# Add edges
graph.add_conditional_edges(
    "agent",
    should_continue,  # Function that decides next step
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "agent")  # After tools, back to agent
```

### 4. Conditional Routing

The agent decides whether to use tools or finish:

```python
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # Go to tools node
    return END          # Finish
```

### 5. Streaming Execution

Watch the agent think in real-time:

```python
for event in agent.stream({"messages": [HumanMessage(content=query)]}):
    for node_name, node_output in event.items():
        # Process each step as it happens
```

## Project Structure

```
react_research_agent/
├── agent.py      # Main agent implementation
└── README.md     # This file
```

## Setup & Run

```bash
# Set API keys
export GOOGLE_API_KEY="your-key"   # https://aistudio.google.com/app/apikey
export TAVILY_API_KEY="your-key"   # https://tavily.com (free tier)

# Install dependencies (from agents/ directory)
pip install -r ../requirements.txt

# Run
python agent.py
```

## Example Output

```
Researching: What are the key features of Python 3.12?
==================================================
[Searching: Python 3.12 key features whats new]
[Processing search results...]
[Searching: PEP 695 PEP 701 Python 3.12 type syntax]
[Processing search results...]

Answer:
Python 3.12 introduced several key features:

1. **Improved Error Messages** - More precise tracebacks
2. **F-string Improvements** (PEP 701) - Nested quotes, comments
3. **Type Parameter Syntax** (PEP 695) - Simpler generics
4. **Per-Interpreter GIL** (PEP 684) - Better parallelism
5. **Comprehension Inlining** (PEP 709) - Faster comprehensions

Sources: docs.python.org, realpython.com
==================================================
```

## Graph Visualization

```
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
            ┌──────►│    Agent     │◄─────┐
            │       │  (LLM Node)  │      │
            │       └──────┬───────┘      │
            │              │              │
            │         tool_calls?         │
            │        /          \         │
            │      Yes           No       │
            │       │             │       │
            │       ▼             ▼       │
            │ ┌──────────┐  ┌──────────┐  │
            │ │  Tools   │  │   END    │  │
            │ │  Node    │  └──────────┘  │
            │ └────┬─────┘                │
            │      │                      │
            └──────┴──────────────────────┘
```

## Key Differences from Calculator Agent

| Calculator Agent | ReAct Research Agent |
|------------------|----------------------|
| Simple while loop | LangGraph state machine |
| Single reasoning step | Multiple iterations |
| Deterministic tools | External API (web search) |
| Direct tool execution | ToolNode abstraction |
| No streaming | Real-time streaming |

## Extending This Agent

### Add More Tools

```python
from langchain_community.tools import WikipediaQueryRun

wiki_tool = WikipediaQueryRun()
tools = [search_tool, wiki_tool]
```

### Add Memory

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)

# Resume conversation
agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "1"}})
```

### Add Human-in-the-Loop

```python
graph.add_node("human_review", human_review_node)
graph.add_edge("tools", "human_review")
graph.add_edge("human_review", "agent")
```

## Tech Stack

| Component | Purpose |
|-----------|---------|
| **LangGraph** | Stateful agent orchestration |
| **LangChain** | LLM abstractions and tools |
| **Google Gemini** | LLM for reasoning |
| **Tavily** | Web search API |

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629) - Original research
- [Tavily API](https://tavily.com) - Search tool
- [LangChain Tools](https://python.langchain.com/docs/integrations/tools/)

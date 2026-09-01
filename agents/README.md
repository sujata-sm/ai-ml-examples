# LLM Agents

Autonomous AI systems that reason and use tools to accomplish tasks.

## Installation

```bash
pip install -r requirements.txt
```

## What are Agents?

LLMs enhanced with the ability to:
- **Plan** - Break down complex tasks
- **Act** - Execute tools/functions
- **Observe** - Process results
- **Iterate** - Refine until complete

## Agent Patterns

| Pattern | Description |
|---------|-------------|
| **Tool Calling** | LLM invokes external functions |
| **ReAct** | Reasoning + Acting loop |
| **Multi-Agent** | Multiple agents collaborate |

## Agent Flow

```
User Query → Think → Act → Observe → Respond
                ↑___________|
                   (loop)
```

## Projects

| Project | Description | Tech Stack |
|---------|-------------|------------|
| [Calculator Agent](./calculator_tool_agent/) | Intro to tool calling | LangChain, Gemini |
| [ReAct Research Agent](./react_research_agent/) | Web search with reasoning | LangGraph, Gemini, Tavily |

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Tools** | Functions the LLM can invoke |
| **State** | Memory persisted across steps |
| **Graph** | Flow control for agent logic |
| **Streaming** | Real-time execution output |

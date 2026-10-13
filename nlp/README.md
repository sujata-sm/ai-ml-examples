# NLP (Natural Language Processing)

Processing and understanding human language with AI.

## What is NLP?
Enabling machines to understand, interpret, and generate human language.

## Projects

| Project | Description | Tech Stack |
|---------|-------------|------------|
| [Sentiment Analysis API](./sentiment_analysis_api/) | REST API for text sentiment analysis | FastAPI, Transformers, DistilBERT |
| Coming Soon | Document Q&A with RAG | LangChain, OpenAI, ChromaDB |

## Techniques

| Technique | Description |
|-----------|-------------|
| **Sentiment Analysis** | Classify text as positive/negative/neutral |
| **RAG** | Retrieval Augmented Generation - ground LLM with documents |
| **Embeddings** | Convert text to dense vectors |
| **Vector Search** | Find similar text using embeddings |

## RAG Flow
```
Documents → Chunking → Embeddings → Vector Store → Retrieve → LLM → Answer
```

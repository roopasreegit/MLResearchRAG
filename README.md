# PaperPro - AI Powered Research Papers Query Engine

# The Problem
Research papers are everywhere, but finding answers across multiple papers is painful. You either:

*Spend hours manually reading through PDFs (slow)
*Use keyword search and get irrelevant results (noisy)
*Copy paste text into ChatGPT and lose source attribution (dangerous)

PaperPro solves this. It understands research context semantically, finds the most relevant papers, and generates answers grounded in real citations.

---

# Features

* Semantic querying over 20+ research papers
* Hybrid Retrieval:

  * BM25 keyword retrieval
  * Dense vector similarity retrieval
* Cohere reranking for relevance optimization
* Query rewriting using LLMs
* Citation-aware answer generation
* ChromaDB vector database with HNSW indexing
* Gemini 2.5 Flash integration
* DeepEval for RAG evaluation
* Streamlit frontend for interactive querying
* Optimized low latency inference pipeline
* Source aware contextual responses

---

# System Architecture

```text
User Query
     ↓
Query Rewriting
     ↓
Hybrid Retrieval
 ├── BM25 Retrieval
 └── Dense Vector Retrieval
     ↓
Document Deduplication
     ↓
Cohere Reranking
     ↓
Context Construction
     ↓
Gemini 2.5 Flash
     ↓
Grounded Answer + Citations
```

---

# Tech Stack

## Backend

* Python
* LangChain

## Retrieval

* ChromaDB
* Sentence Transformers
* BM25 Retriever

## LLMs

* Gemini 2.5 Flash
* Cohere Rerank API

## Frontend

* Streamlit

## Evaluation

* DeepEval

---

# Core Components

# 1. Hybrid Retrieval

PaperPro uses a hybrid retrieval pipeline combining:

## BM25 Retrieval

Captures:

* exact keyword matches
* acronyms
* paper-specific terminology
* lexical similarity

## Dense Retrieval

Uses Sentence Transformers embeddings to capture:

* semantic meaning
* paraphrased queries
* contextual similarity

This improves retrieval quality significantly over standalone retrieval methods.

---

# 2. ChromaDB Vector Store

Dense embeddings are stored inside ChromaDB using:

* persistent vector storage
* HNSW indexing
* efficient similarity search

Benefits:

* fast retrieval
* scalable vector search
* persistent storage

---

# 3. Query Rewriting

Before retrieval, ambiguous or vague user queries are rewritten using Gemini.

Example:

Input:

```text
How can models become more reliable?
```

Rewritten:

```text
What techniques improve factual reliability in large language models during inference and deployment?
```

This improves:

* retrieval relevance
* semantic precision
* reranking quality

---

# 4. Cohere Reranking

Retrieved documents are reranked using Cohere Rerank API.

Purpose:

* reorder retrieved chunks by semantic relevance
* filter noisy retrievals
* improve context quality before generation

This significantly improves final answer quality.

---

# 5. Answer Generation

The final response is generated using:

* Gemini 2.5 Flash
* context-aware prompting
* source-grounded synthesis

The model:

* synthesizes across multiple papers
* cites relevant sources
* avoids hallucination when context is insufficient

---

# Performance Optimization

Initial end-to-end query latency:

```text
~2 minutes
```

Optimized latency:

```text
~9 seconds
```

## Optimizations Performed

### Cached ChromaDB Loading

Used:

```python
@st.cache_resource
```

to avoid reloading vector stores on every rerun.

### Cached BM25 Retriever

Prevented reconstruction of BM25 indexes for each query.

### Cached Metadata

Cached paper metadata and sidebar information to reduce Streamlit reruns.

### Reduced Redundant LLM Calls

Optimized retrieval flow to minimize unnecessary generation overhead.

### Persistent Model Initialization

Models are initialized once and reused across requests.

---

# Evaluation Pipeline

PaperPro includes a DeepEval-based evaluation framework.

## Metrics Evaluated

### Answer Relevancy

Measures how well generated answers address the query.

### Faithfulness

Measures whether generated responses remain grounded in retrieved context.

### Contextual Recall

Measures whether the retrieval system fetched sufficient relevant information.

---

# Example Query
<img width="1781" height="870" alt="Screenshot 2026-05-17 084610" src="https://github.com/user-attachments/assets/d5ee70d0-ddf3-480f-a704-c22176552210" />



# Installation

# 1. Clone Repository

```bash
git clone https://github.com/yourusername/MLResearchRAG.git

cd MLResearchRAG
```

---

# 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

## Windows

```bash
venv\Scripts\activate
```

## Linux/Mac

```bash
source venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Configure Environment Variables

Create `.env`

```env
GEMINI_API_KEY=your_key
COHERE_API_KEY=your_key
```

---

# 5. Run Application

```bash
streamlit run app.py
```

---

# Future Improvements

* Multi-paper comparative reasoning
* Conversational memory
* PDF upload support
* Research graph visualization
* Agentic retrieval workflows
* Citation export
* Multi-modal paper understanding
* RAGAS evaluation integration
* Streaming token generation
* Redis caching layer
* GPU inference optimization

---

# Deployment

The application is deployed using:

* Streamlit Cloud

Can also be deployed on:

* HuggingFace Spaces
* Render
* Railway
* AWS/GCP/Azure

---

# Key Learnings

Through this project:

* implemented production-style RAG architectures
* optimized low-latency retrieval systems
* worked with vector databases and semantic search
* designed evaluation pipelines for generative AI systems
* improved retrieval grounding and hallucination reduction
* built scalable AI-assisted research workflows

---

# License

MIT License

---

# Author

Roopasree
Computer Science Engineering, NIT Trichy

Made with <3 for Researchers


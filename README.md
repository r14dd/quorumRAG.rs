<!-- ⚠️ This README has been generated from the file(s) "blueprint.md" ⚠️-->

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-quorumrag)

# ➤ QuorumRAG
### **Multi-Retriever Retrieval-Augmented Generation via Quorum Consensus**

**Query → Multi-Retriever Ensemble → RRF Scoring → Quorum Filtering → Evidence Clustering → LLM Generation**

A research implementation of QuorumRAG — a RAG architecture that requires cross-retriever consensus before surfacing evidence, built entirely in Rust with Ollama for local LLM inference.

<p align="center">
  <img src="https://img.shields.io/badge/Rust-111827?style=for-the-badge&logo=rust&logoColor=CE422B" alt="Rust" />
  <img src="https://img.shields.io/badge/Ollama-111827?style=for-the-badge&logo=ollama&logoColor=FFFFFF" alt="Ollama" />
  <img src="https://img.shields.io/badge/Tantivy-111827?style=for-the-badge&logo=rust&logoColor=CE422B" alt="Tantivy" />
  <img src="https://img.shields.io/badge/Tokio-111827?style=for-the-badge&logo=rust&logoColor=CE422B" alt="Tokio" />
</p>

---

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-what-is-quorumrag)

## ➤ ⚡ What is QuorumRAG?

QuorumRAG is a retrieval strategy that runs multiple independent retrievers (dense semantic search at different chunk granularities + BM25 keyword search) over the same corpus, then only surfaces evidence that achieves **quorum** — agreement from at least N retrievers. Evidence clusters are scored using **Reciprocal Rank Fusion (RRF)** before being passed to the LLM, producing answers grounded in cross-validated evidence rather than the output of a single retriever.

---

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-why-it-stands-out)

## ➤ ✨ Why it stands out

- **Quorum consensus** — only evidence agreed upon by multiple retrievers reaches the LLM
- **RRF scoring** — rank-based fusion robust to BM25/cosine scale mismatch
- **Overlapping chunks** — 50% stride prevents answers being split at chunk boundaries
- **Parallel embedding** — batched concurrent requests for fast corpus indexing
- **Embedding cache** — cold start only happens once per retriever configuration
- **Full eval harness** — baseline vs. QuorumRAG recall comparison on every run
- **Built entirely in Rust** — no Python runtime, single binary, production-grade performance

---

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-design-decisions)

## ➤ 🧠 Design Decisions

| Decision | Why it matters |
|---|---|
| **Reciprocal Rank Fusion** | Normalizes scores across retrievers without manual scaling — 1/(k+rank) is robust and well-established (Cormack et al., 2009) |
| **Quorum filtering** | Reduces hallucination risk by requiring cross-retriever agreement before evidence reaches the LLM |
| **Multi-granularity dense retrieval** | Dense-50, Dense-100, Dense-200 capture answers at different levels of context — fine detail to broad context |
| **BM25 as a quorum voter** | Keyword retrieval as a complementary signal to semantic search; if both agree, confidence is higher |
| **Overlapping chunks (50% stride)** | Answers near chunk boundaries are captured by at least one window |
| **Embedding cache per retriever** | Avoids re-embedding thousands of chunks on every run; cache is keyed by retriever ID including chunk size and overlap |

---

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-architecture)

## ➤ 🏗️ Architecture

```
Query
  │
  ├─► Dense-50   (ov25)  ─┐
  ├─► Dense-100  (ov50)  ─┤  RRF Scoring
  ├─► Dense-200  (ov100) ─┤  (1 / k + rank)
  └─► BM25-100   (ov50)  ─┘
              │
              ▼
     Embedding Similarity
        Clustering (0.85)
              │
              ▼
     Quorum Filter
     (support ≥ 2 retrievers)
              │
              ▼
     Rank Clusters
     (0.7 × avg_score + 0.3 × support)
              │
              ▼
     Build Context (top 5 clusters,
     all members deduplicated by score)
              │
              ▼
     Ollama LLM Generation
```

---

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-pipeline-modules)

## ➤ 🧩 Pipeline Modules

| Module | Purpose |
|---|---|
| `corpus` | Loads `.txt` files, chunks with configurable size and overlap |
| `embedding` | HTTP client for Ollama `nomic-embed-text` embeddings |
| `retrievers/dense` | Cosine similarity search over embedded chunks |
| `retrievers/bm25` | Tantivy-powered BM25 keyword search |
| `clustering` | Greedy cosine similarity clustering of candidates |
| `quorum` | Filters clusters below the minimum retriever support threshold |
| `ranking` | Scores clusters by RRF avg + support weighting |
| `context` | Builds the LLM context string from top-ranked clusters |
| `generation` | Ollama generation API client |
| `evaluation` | Word-overlap recall metric, baseline vs. QuorumRAG comparison |

---

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-quickstart)

## ➤ 🚀 Quickstart

### Prerequisites

- [Rust](https://rustup.rs/) (edition 2024)
- [Ollama](https://ollama.com/) running locally at `http://localhost:11434`
- Required models pulled:

```bash
ollama pull nomic-embed-text
ollama pull mistral
```

### 1) Fetch the corpus

```bash
pip install wikipedia-api
python3 scripts/fetch_corpus.py
```

### 2) Run eval (builds embedding cache on first run)

```bash
cargo run
```

### 3) Ask a single question

```bash
cargo run -- --query "What is backpropagation?"
```

---

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-configuration)

## ➤ ⚙️ Configuration

`config.toml` controls the full pipeline:

```toml
quorum_threshold = 2      # minimum retrievers that must agree
top_k = 15                # candidates per retriever per query
cluster_threshold = 0.85  # cosine similarity threshold for clustering

[[retrievers]]
retriever_type = "dense"
chunk_size = 50
overlap = 25

[[retrievers]]
retriever_type = "dense"
chunk_size = 100
overlap = 50

[[retrievers]]
retriever_type = "dense"
chunk_size = 200
overlap = 100

[[retrievers]]
retriever_type = "bm25"
chunk_size = 100
overlap = 50

[ollama]
url = "http://localhost:11434"
model = "mistral"
```

---

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-eval-results)

## ➤ 📊 Eval Results

Evaluated on 20 Wikipedia-based Q&A pairs across CS and ML topics.

| System | Recall |
|---|---|
| Baseline (Dense-50 only) | 14 / 20 |
| **QuorumRAG (4 retrievers)** | **19 / 20** |

QuorumRAG additionally provides a **support score** (1–4) on every answer, indicating how many independent retrievers agreed on the evidence — a confidence signal the baseline cannot produce.

---

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-tech-stack)

## ➤ 🧠 Tech Stack

- **Rust** (edition 2024) — entire implementation
- **Tokio** — async runtime, parallel embedding
- **Tantivy** — BM25 full-text search index
- **Ollama** — local LLM inference (`nomic-embed-text`, `mistral`)
- **Serde / TOML** — config and cache serialization
- **Futures** — batched concurrent HTTP embedding

---

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-whats-next)

## ➤ 🛣️ What's next

- Confidence-weighted answer generation using support scores
- Streaming LLM responses for interactive mode
- PyO3 bindings to expose the Rust core to a Python research harness
- Additional retrievers (TF-IDF, hybrid sparse-dense)
- Ablation study tooling (sweep quorum threshold, chunk size, cluster threshold)
- REST API mode for integration with external frontends

---

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#-license)

## ➤ Authors

- [Riad Mukhtarov](https://www.linkedin.com/in/riadmukhtarov/)

---

## ➤ License

[MIT](https://choosealicense.com/licenses/mit/)

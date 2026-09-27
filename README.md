# RBI FAQ Advisor (AgenticProject)

An agentic RAG system that answers policy questions over the RBI's *Report on Currency and Finance 2024–25* — with a built-in evaluation harness to measure retrieval and generation quality, not just eyeball the outputs.

This project doubles as a hands-on exploration of **RAG evaluation and governance**: every answer is traceable back to source chunks, and every pipeline change can be scored against a golden test set using [DeepEval](https://github.com/confident-ai/deepeval).

---

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Report_2024-25  │────▶│   Docling    │────▶│  Markdown Chunker │────▶│  Chroma DB    │
│      .pdf        │     │ (PDF → MD)   │     │ (batch-tagged)    │     │ (embeddings)  │
└─────────────────┘     └──────────────┘     └─────────────────┘     └──────┬───────┘
                                                                              │
┌──────────────┐     ┌──────────────┐     ┌──────────────┐                  │
│  User Query   │────▶│  Retriever    │◀────────────────────────────────────┘
└──────────────┘     │ (bi-encoder,  │
                      │  k=3 similarity) │
                      └──────┬───────┘
                             │ context
                             ▼
                      ┌──────────────┐
                      │  RAG Chain    │  ChatGroq (openai/gpt-oss-120b)
                      │  (LCEL)       │  "Answer using only the context"
                      └──────┬───────┘
                             │ answer + retrieval_context
                             ▼
                      ┌──────────────┐
                      │  DeepEval     │  GroqJudge (openai/gpt-oss-20b)
                      │  Metrics      │  Contextual Precision / Recall
                      └──────────────┘
```

**Ingestion → Vector store** happens once, offline, via `main.py`.
**Retrieval → Generation** happens per query, via `RAGChain`.
**Evaluation** runs the RAG chain against a golden set and scores it, via `Evaluate.py`.

---

## Project structure

```
AgenticProject/
├── main.py                       # Entry point: run the full ingestion pipeline
├── config.json                   # Global + per-strategy chunking config
├── Documents/
│   └── Report_2024-25.pdf        # Source document (RBI report)
├── Utils/
│   ├── config_loader.py          # Merges global + strategy config
│   └── rag_settings.py           # Active chunking strategy switch
├── rbiFaqAgent/
│   ├── Data/
│   │   └── final_report_output.md  # Docling-parsed markdown, batch-tagged
│   ├── RAG/
│   │   ├── IngestDocuments.py    # PDF → Markdown via Docling (batched, memory-safe)
│   │   ├── MarkDown_chunking.py  # Markdown → LangChain Documents (+ batch metadata)
│   │   ├── vector_Store.py       # Batched writes into Chroma
│   │   ├── Retriever.py          # Bi-encoder similarity retriever (k=3)
│   │   ├── RAG_chain.py          # LCEL retrieval + generation chain
│   │   └── Vector_Audit.py       # Sanity-checks the persisted vector DB
│   └── Evals/
│       ├── Goldens/goldens.json  # Golden Q&A test set
│       ├── GroqJudge.py          # Custom DeepEval judge (ChatGroq wrapper)
│       ├── EvaluationJudge.py    # Judge model provider
│       ├── metrics.py            # DeepEval metrics configuration
│       ├── Evaluate.py           # Runs goldens through the chain, scores results
│       └── Results/              # Timestamped eval run outputs (JSON)
└── pyproject.toml
```

---

## Setup

**Requirements:** Python ≥ 3.10, a [Groq](https://console.groq.com/) API key.

```bash
# Install dependencies (using uv)
uv sync

# Or with pip
pip install -e .
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

---

## Usage

### 1. Parse the source PDF into Markdown

Docling parsing is run in page-range batches to stay memory-safe on constrained hardware. Adjust `START_PAGE` / `END_PAGE` in `IngestDocuments.py`, then:

```bash
python -m rbiFaqAgent.RAG.IngestDocuments
```

This appends parsed batches to `rbiFaqAgent/Data/final_report_output.md`, tagging each with a `<!-- PDF BATCH n-m -->` marker (page-level provenance from Docling wasn't reliably preserved across batches, so traceability is at the batch level rather than the exact page).

### 2. Build the vector store

```bash
python main.py
```

This chunks the markdown (per the active strategy in `config.json`), embeds it with `all-MiniLM-L6-v2`, writes it into a local Chroma collection, and runs a quick audit of what got persisted.

### 3. Query the RAG chain

```python
from rbiFaqAgent.RAG.RAG_chain import RAGChain

chain = RAGChain()
result = chain.invoke_rag_chain("What are India's stablecoin regulations?")
print(result["answer"])
```

### 4. Run the evaluation suite

```bash
python -m rbiFaqAgent.Evals.Evaluate
```

Loads goldens → runs each through the live RAG chain → scores the result with DeepEval metrics → writes a timestamped report to `rbiFaqAgent/Evals/Results/`.

---

## Configuration

`config.json` defines a global block plus swappable chunking strategies:

```json
{
  "global_settings": {
    "embedding_model_name": "all-MiniLM-L6-v2",
    "batch_size": 50
  },
  "strategies": {
    "markdown_fixed_size": { "chunk_size": 1000, "chunk_overlap": 200 },
    "semantic_chunking": { "breakpoint_threshold_type": "percentile", "buffer_size": 1 },
    "recursive_character": { "chunk_size": 500, "chunk_overlap": 50 }
  }
}
```

The active strategy is set in `Utils/rag_settings.py` (`chunking_strategy = "markdown_fixed_size"`). Only the markdown strategy has a working splitter implementation today; `semantic_chunking` and `recursive_character` are reserved for future comparison.

---

## Evaluation

Two of DeepEval's RAG metrics are currently active, both judged by a self-hosted `ChatGroq` judge model rather than an external API:

| Metric | Threshold | What it measures |
|---|---|---|
| Contextual Recall | 0.7 | Did retrieval surface everything needed to answer the question? |
| Contextual Precision | 0.7 | Are the retrieved chunks ranked with the most relevant ones first? |

Faithfulness and Answer Relevancy are defined in `metrics.py` but not yet enabled — planned next.

### A finding worth calling out

A real eval run on the question *"What are the Regulations for Stablecoins in Singapore?"* surfaced a retrieval ranking issue:

- **Contextual Recall: 1.0** — the correct fact (MAS 2023 framework) was retrieved.
- **Contextual Precision: 0.33** — it was ranked **3rd of 3** retrieved chunks, behind two irrelevant ones (US and Japan stablecoin rules).
- **The final answer was still correct** — the LLM was able to find the right fact despite the noisy ranking.

This is a useful example of **generation robustness masking a retrieval quality problem**: with a smaller `k`, a weaker LLM, or a noisier corpus, this same ranking issue could have produced a wrong or incomplete answer. It's exactly the kind of gap that end-to-end "did it get the right answer" testing misses, and that retrieval-specific metrics like Contextual Precision are designed to catch.

---

## Known limitations / Roadmap

- [ ] Golden set currently has a single test case — needs expansion across more sections of the report for meaningful trend tracking
- [ ] Faithfulness and Answer Relevancy metrics are implemented but disabled
- [ ] `semantic_chunking` and `recursive_character` strategies are configured but not yet wired into the chunking code
- [ ] Page-level citation isn't available (batch-level only) due to Docling provenance limitations across split PDF batches
- [ ] Two different Chroma integration packages (`langchain_community` and `langchain_chroma`) are used in different files and should be consolidated

---

## Tech stack

- **Parsing:** Docling
- **Orchestration:** LangChain (LCEL)
- **LLM:** Groq (`openai/gpt-oss-120b` for generation, `openai/gpt-oss-20b` as judge)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store:** Chroma
- **Evaluation:** DeepEval
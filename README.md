# Operation Ledger-Mind: The Financial Intelligence

A comparison of Parametric Memory (Fine-Tuning) vs Non-Parametric Memory (RAG) for financial document analysis using Uber's 2024 Annual Report.

## Project Overview

This project implements and evaluates two approaches to financial Q&A on the same base model (Llama-3.2-3B-Instruct):

1. **Fine-tuned (Parametric Memory)**: the base model fine-tuned on generated Q&A pairs with QLoRA
2. **RAG (Non-Parametric Memory)**: the same base model answering from a hybrid retrieval pipeline on Weaviate

## Project Structure

```
operation_ledger_mind/
├── notebooks/
│   ├── 01_data_factory.ipynb          # Chunking + Q&A dataset generation
│   ├── 02_finetuning.ipynb       # QLoRA fine-tuning (Colab/Kaggle T4)
│   ├── 03_rag.ipynb                   # Hybrid RAG index
│   ├── 04_evaluation.ipynb            # Fine-tuned vs RAG evaluation
│   ├── inference_utils.py             # Loads both systems on one base model
│   └── artifacts/
│       ├── chunks.jsonl               # Chunks from notebook 01, indexed by notebook 03
│       ├── data_factory/
│       │   ├── raw_pairs.jsonl
│       │   ├── train.jsonl
│       │   └── golden_test_set.jsonl
│       └── adapters/                  # LoRA adapters from notebook 02
├── src/
│   ├── config/
│   │   └── config.yaml                # Configuration file
│   └── services/
│       └── llm_services.py            # LLM and embedding services
├── data/
│   └── 2024-Annual-Report.pdf         # Uber's annual report (place here)
├── pyproject.toml
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
cd operation_ledger_mind

# Install dependencies
uv sync

# Create .env file with API keys
cat > .env << EOF
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_huggingface_token
EOF
```

### 2. Add Data

Place Uber's 2024 Annual Report PDF in the `data/` directory:
```bash
cp /path/to/uber-2024-annual-report.pdf data/2024-Annual-Report.pdf
```

### 3. Run Notebooks

Execute the notebooks in order:

1. `01_data_factory.ipynb`: generate the Q&A dataset (Groq API)
2. `02_finetuning.ipynb`: fine-tune on a T4 (Colab or Kaggle), then copy the adapters to `notebooks/artifacts/adapters/`
3. `03_rag.ipynb`: build the Weaviate index (Linux or Mac; embedded Weaviate does not run on Windows)
4. `04_evaluation.ipynb`: evaluate and compare

## Key Features

### Notebook 01: Data Factory
- **PDF Processing**: Extracts text per page and tables as row chunks with their headings
- **Q&A Generation**: Up to 3 Q&A pairs per chunk with `openai/gpt-oss-120b` on Groq
- **Resumable**: Saves after every chunk and stops cleanly on the daily rate limit
- **Output**: `train.jsonl` (80%) and `golden_test_set.jsonl` (20%), split by chunk

### Notebook 02: Fine-Tuning
- **Model**: `unsloth/Llama-3.2-3B-Instruct`
- **Method**: QLoRA (4-bit NF4 base, FP16 compute on T4)
- **Adapters**: LoRA (r=16, alpha=32) on the attention projections
- **Training**: TRL `SFTTrainer`, 3 epochs
- **Output**: LoRA adapters

### Notebook 03: RAG
- **Vector DB**: Weaviate Embedded
- **Hybrid Search**: Dense vectors (embeddings) + BM25 (keyword)
- **Fusion**: Reciprocal Rank Fusion (RRF)
- **Reranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2)

### Notebook 04: Evaluation
- **Fair setup**: both systems use the same base model, system prompt and greedy decoding; RAG runs with the adapters disabled
- **Metrics**:
  - ROUGE-L: Textual overlap
  - LLM-as-Judge: Accuracy (1-5 scale), `openai/gpt-oss-120b` at temperature 0
  - Latency: Response time in milliseconds
  - Retrieval hit rate: whether RAG retrieved the chunk each test question came from
- **Cost Analysis**: GPU serving cost, plus the vector DB for RAG

## Technologies Used

- **LLMs**: Llama-3.2-3B-Instruct, gpt-oss-120b (via Groq)
- **Fine-Tuning**: `transformers`, `peft`, `trl`, `bitsandbytes`
- **RAG**: Weaviate, sentence-transformers
- **Evaluation**: rouge-score, LLM-as-Judge
- **Utilities**: pandas, matplotlib

## Configuration

Edit `src/config/config.yaml` to customize:
- LLM provider and models (`llm_model`, `data_factory_model`, `finetune_model`)
- Embedding model
- Adapter folder name (`adapter_name`)

## Requirements

- Python 3.10+
- Groq API key (data factory and judge)
- **Notebook 02**: NVIDIA GPU, e.g. a free Colab or Kaggle T4
- **Notebooks 03-04**: Linux or Mac (embedded Weaviate); notebook 04 runs on CUDA, Apple MPS or CPU

## Acknowledgments

- **Uber Technologies**: For the [2024 Annual Report](https://s23.q4cdn.com/407969754/files/doc_financials/2025/Stockholders2025/2024-Annual-Report.pdf)
- **Hugging Face**: For model hosting and libraries
- **Weaviate**: For vector database technology

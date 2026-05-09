# Operation Ledger-Mind: The Financial Intelligence

**AI Engineer Essentials - Mini Project 01**

A comprehensive comparison of Parametric Memory (Fine-Tuning) vs Non-Parametric Memory (RAG) for financial document analysis using Uber's 2024 Annual Report.

## 🎯 Project Overview

This project implements and evaluates two AI architectures for financial Q&A:

1. **"The Intern" (Parametric Memory)**: Fine-tuned Llama-3-8B model with QLoRA
2. **"The Librarian" (Non-Parametric Memory)**: Advanced Hybrid RAG system with Weaviate

## 📁 Project Structure

```
operation_ledger_mind/
├── notebooks/
│   ├── 01_data_factory.ipynb          # Q&A dataset generation
│   ├── 02_finetuning_intern.ipynb     # Fine-tuning with QLoRA
│   ├── 03_rag_librarian.ipynb         # Hybrid RAG system
│   └── 04_evaluation_arena.ipynb      # Comprehensive evaluation
├── src/
│   ├── config/
│   │   └── config.yaml                # Configuration file
│   ├── services/
│   │   └── llm_services.py            # LLM and embedding services
│   └── utils/
│       ├── pdf_utils.py               # PDF processing utilities
│       └── eval_utils.py              # Evaluation utilities
├── data/
│   └── 2024-Annual-Report.pdf         # Uber's annual report (place here)
├── artifacts/                         # Generated artifacts
│   ├── data_factory/
│   │   ├── train.jsonl
│   │   ├── golden_test_set.jsonl
│   │   └── chunks.txt
│   ├── finetuned_model/               # Fine-tuned adapters
│   └── evaluation/                    # Evaluation results
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or download the project
cd operation_ledger_mind

# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys
cat > .env << EOF
OPENROUTER_API_KEY=your_openrouter_key
HF_TOKEN=your_huggingface_token
# Optional: for Weaviate Cloud
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_key
EOF
```

### 2. Add Data

Place Uber's 2024 Annual Report PDF in the `data/` directory:
```bash
cp /path/to/uber-2024-annual-report.pdf data/2024-Annual-Report.pdf
```

### 3. Run Notebooks

Execute the notebooks in order:

```bash
# 1. Generate Q&A dataset
jupyter notebook notebooks/01_data_factory.ipynb

# 2. Fine-tune model (requires GPU)
jupyter notebook notebooks/02_finetuning_intern.ipynb

# 3. Build RAG system
jupyter notebook notebooks/03_rag_librarian.ipynb

# 4. Evaluate and compare
jupyter notebook notebooks/04_evaluation_arena.ipynb
```

## 📊 Key Features

### Notebook 01: Data Factory
- **PDF Processing**: Extracts and chunks Uber's annual report
- **Dual-LLM Pipeline**: Generates 10 Q&A pairs per chunk
- **Question Categories**: Hard Facts, Strategic Summaries, Stylistic/Creative
- **Output**: `train.jsonl` (80%) and `golden_test_set.jsonl` (20%)

### Notebook 02: The Intern (Fine-Tuning)
- **Model**: Llama-3-8B-Instruct
- **Quantization**: 4-bit NF4 with double quantization
- **Adapters**: LoRA (r=16, alpha=32)
- **Training**: SFTTrainer with ~100 steps
- **Output**: Trained adapters + `query_intern()` function

### Notebook 03: The Librarian (RAG)
- **Vector DB**: Weaviate (Embedded or Cloud)
- **Hybrid Search**: Dense vectors (embeddings) + BM25 (keyword)
- **Fusion**: Reciprocal Rank Fusion (RRF)
- **Reranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Output**: `query_librarian()` function

### Notebook 04: Evaluation Arena
- **Metrics**:
  - ROUGE-L: Textual overlap
  - LLM-as-Judge: Faithfulness & Accuracy (1-5 scale)
  - Latency: Response time in milliseconds
- **Hallucination Audit**: Analyzes number accuracy
- **Cost Analysis**: AWS GPU vs API costs for 150k queries/month

## 🎓 Learning Outcomes

1. **Data Generation**: Create instruction datasets from raw documents
2. **Fine-Tuning**: QLoRA for memory-efficient model adaptation
3. **Advanced RAG**: Hybrid search with reranking for high precision
4. **Evaluation**: Multi-metric assessment including cost-benefit

## 📈 Expected Results

The evaluation will reveal:
- **Accuracy**: Which system is more factually correct?
- **Faithfulness**: Which stays true to the source material?
- **Latency**: Which responds faster?
- **Cost**: Which is more economical at scale?

## 🛠️ Technologies Used

- **LLMs**: Llama-3, GPT-4o-mini (via OpenRouter)
- **Fine-Tuning**: Hugging Face (`transformers`, `peft`, `trl`, `bitsandbytes`)
- **RAG**: Weaviate, sentence-transformers, rank-bm25
- **Evaluation**: rouge-score, LLM-as-Judge
- **Utilities**: pandas, matplotlib, seaborn

## ⚙️ Configuration

Edit `src/config/config.yaml` to customize:
- LLM provider and model
- Embedding model
- Chunk size and overlap
- Fine-tuning hyperparameters
- RAG retrieval parameters

## 📝 Requirements

### For All Notebooks:
- Python 3.9+
- API access to LLM provider (OpenRouter recommended)
- HuggingFace account (for model downloads)

### For Notebook 02 (Fine-Tuning):
- GPU with 16GB+ VRAM (e.g., T4, A10G, RTX 4090)
- Google Colab Pro or similar (if not local GPU)

### For Notebook 03 (RAG):
- Weaviate Embedded OR Weaviate Cloud account

## 🏆 Submission Checklist

- [ ] All 4 notebooks executed and tested
- [ ] `train.jsonl` and `golden_test_set.jsonl` generated
- [ ] Fine-tuned adapters saved
- [ ] Evaluation results with visualizations
- [ ] Engineering Report (PDF, 1500 words)
- [ ] Full source code with folder structure preserved

## 📄 License

This project is for educational purposes as part of the AI Engineer Essentials course.

## 🤝 Acknowledgments

- **Uber Technologies**: For the 2024 Annual Report
- **Hugging Face**: For model hosting and libraries
- **Weaviate**: For vector database technology
- **Anthropic/OpenAI**: For LLM APIs

---

**Note**: This README assumes you're running in a Colab/Jupyter environment with appropriate GPU access for fine-tuning.

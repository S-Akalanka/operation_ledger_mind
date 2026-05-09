#!/usr/bin/env python
"""
Verify all RAG Systems dependencies are installed correctly.
Run this after installing requirements.txt to ensure everything works.
"""

import sys

def test_imports():
    """Test all critical package imports."""
    
    missing = []
    working = []
    
    # Test each package group
    tests = {
        "Core Data Processing": [
            ("pandas", "import pandas"),
            ("numpy", "import numpy"),
            ("PIL (pillow)", "from PIL import Image"),
            ("matplotlib", "import matplotlib"),
            ("scikit-learn", "from sklearn.feature_extraction.text import TfidfVectorizer"),
            ("pyarrow", "import pyarrow"),
        ],
        "ML & Embeddings": [
            ("sentence-transformers", "from sentence_transformers import SentenceTransformer"),
            ("torch", "import torch"),
        ],
        "Vector Database": [
            ("chromadb", "import chromadb"),
        ],
        "LangChain Core": [
            ("langchain", "import langchain"),
            ("langchain-core", "from langchain_core import prompts"),
            ("langchain-community", "from langchain_community.embeddings import HuggingFaceEmbeddings"),
            ("langchain-text-splitters", "from langchain_text_splitters import RecursiveCharacterTextSplitter"),
        ],
        "LangChain Providers": [
            ("langchain-openai", "from langchain_openai import ChatOpenAI, OpenAIEmbeddings"),
            ("langchain-groq", "from langchain_groq import ChatGroq"),
            ("langchain-google-genai", "from langchain_google_genai import ChatGoogleGenerativeAI"),
            ("langchain-cohere", "from langchain_cohere import CohereEmbeddings"),
            ("langchain-chroma", "from langchain_chroma import Chroma"),
        ],
        "LlamaIndex Core": [
            ("llama-index", "from llama_index.core import VectorStoreIndex, Settings"),
        ],
        "LlamaIndex Providers": [
            ("llama-index-llms-openai", "from llama_index.llms.openai import OpenAI"),
            ("llama-index-llms-groq", "from llama_index.llms.groq import Groq"),
            ("llama-index-llms-gemini", "from llama_index.llms.gemini import Gemini"),
            ("llama-index-embeddings-openai", "from llama_index.embeddings.openai import OpenAIEmbedding"),
            ("llama-index-embeddings-cohere", "from llama_index.embeddings.cohere import CohereEmbedding"),
            ("llama-index-embeddings-huggingface", "from llama_index.embeddings.huggingface import HuggingFaceEmbedding"),
            ("llama-index-vector-stores-chroma", "from llama_index.vector_stores.chroma import ChromaVectorStore"),
        ],
        "API Clients": [
            ("openai", "import openai"),
            ("google-generativeai", "import google.generativeai"),
            ("cohere", "import cohere"),
        ],
        "Document Processing": [
            ("pypdf", "from pypdf import PdfReader"),
        ],
        "Sparse Retrieval": [
            ("rank-bm25", "from rank_bm25 import BM25Okapi"),
        ],
        "Environment": [
            ("python-dotenv", "from dotenv import load_dotenv"),
        ],
        "Jupyter": [
            ("jupyter", "import jupyter_core"),
            ("ipykernel", "import ipykernel"),
            ("ipywidgets", "import ipywidgets"),
            ("jupyterlab", "import jupyterlab"),
        ],
    }
    
    print("🔍 Verifying RAG Systems Dependencies...\n")
    print("=" * 70)
    
    for category, imports in tests.items():
        print(f"\n📦 {category}")
        print("-" * 70)
        
        for package_name, import_statement in imports:
            try:
                exec(import_statement)
                print(f"  ✅ {package_name}")
                working.append(package_name)
            except ImportError as e:
                print(f"  ❌ {package_name} - {e}")
                missing.append(package_name)
    
    print("\n" + "=" * 70)
    print(f"\n📊 Summary:")
    print(f"  ✅ Working: {len(working)}/{len(working) + len(missing)}")
    
    if missing:
        print(f"  ❌ Missing: {len(missing)}")
        print(f"\n❌ Missing packages:")
        for pkg in missing:
            print(f"     - {pkg}")
        print(f"\n💡 Install missing packages:")
        print(f"     uv pip install -r requirements.txt")
        return False
    else:
        print(f"\n🎉 All dependencies installed successfully!")
        print(f"\n✨ You're ready to run the notebooks!")
        print(f"\nNext steps:")
        print(f"  1. Copy .env.example to .env and add your API keys")
        print(f"  2. Start Jupyter: jupyter lab")
        print(f"  3. Open notebooks in order (01, 10, 11, 12, 20, 30)")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)


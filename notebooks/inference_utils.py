
from src.services.llm_services import load_config

config = load_config("../src/config/config.yaml")

# ============================================================
# Load The Intern (Fine-Tuned Model)
# ============================================================

print("Loading The Intern (Fine-Tuned Model)...")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

# Model config
model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Same as Notebook 02
adapter_path = Path(config['artifacts_root']) / "finetuned_model" / "final_adapters"

# Check if adapters exist
if not adapter_path.exists():
    print(f"❌ Adapters not found at {adapter_path}")
    print("   Please run Notebook 02 first!")


    # Placeholder function
    def query_intern(question):
        return "[Error: Run Notebook 02 first to create fine-tuned model]"


    print("⚠️  Using placeholder query_intern()")

else:
    print(f"Loading from: {adapter_path}")

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        dtype=torch.float16,
        attn_implementation="eager"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("⏳ Loading fine-tuned adapters...")
    # Load fine-tuned adapters
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("✅ Model loaded!")


    # Define inference function
    def query_intern(question, max_new_tokens=256):
        """Query the fine-tuned Intern model."""
        messages = [
            {
                "role": "system",
                "content": "You are a financial analyst trained on Uber's 2024 Annual Report."
            },
            {"role": "user", "content": question}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        return response


    print("✅ query_intern() ready!")

# ============================================================
# Load The Librarian (RAG System)
# ============================================================

print("Loading The Librarian (RAG System)...")

import weaviate
from sentence_transformers import CrossEncoder

# Initialize LLM and embeddings
from src.services.llm_services import get_llm, get_text_embeddings

llm = get_llm(config)
embeddings = get_text_embeddings(config)

print("✅ LLM and embeddings initialized")

# Connect to Weaviate
try:
    print("Starting Weaviate with extended startup period...")
    client = weaviate.connect_to_embedded(version="latest")

    collection_name = "UberFinancials"

    # Check if collection exists
    if not client.collections.exists(collection_name):
        print(f"❌ Collection '{collection_name}' not found")
        print("   Please run Notebook 03 first!")
        raise Exception("RAG collection not found")

    print(f"✅ Connected to collection: {collection_name}")

    # Load reranker
    print("⏳ Loading cross-encoder reranker...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("✅ Reranker loaded")


    # Define search functions
    def dense_search(query, top_k=10):
        """Dense vector search."""
        query_vector = embeddings.embed_query(query)
        collection = client.collections.get(collection_name)

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=['distance']
        )

        results = []
        for obj in response.objects:
            results.append({
                'chunk_id': obj.properties['chunk_id'],
                'text': obj.properties['text'],
                'score': 1 - obj.metadata.distance
            })
        return results


    def bm25_search(query, top_k=10):
        """BM25 keyword search."""
        collection = client.collections.get(collection_name)
        response = collection.query.bm25(query=query, limit=top_k)

        results = []
        for obj in response.objects:
            results.append({
                'chunk_id': obj.properties['chunk_id'],
                'text': obj.properties['text'],
                'score': obj.metadata.score if obj.metadata.score else 0.0
            })
        return results


    def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
        """Combine dense and BM25 results using RRF."""
        rrf_scores = {}

        # Add dense results
        for rank, result in enumerate(dense_results, 1):
            cid = result['chunk_id']
            if cid not in rrf_scores:
                rrf_scores[cid] = {
                    'text': result['text'],
                    'chunk_id': cid,
                    'rrf_score': 0.0
                }
            rrf_scores[cid]['rrf_score'] += 1 / (k + rank)

        # Add BM25 results
        for rank, result in enumerate(bm25_results, 1):
            cid = result['chunk_id']
            if cid not in rrf_scores:
                rrf_scores[cid] = {
                    'text': result['text'],
                    'chunk_id': cid,
                    'rrf_score': 0.0
                }
            rrf_scores[cid]['rrf_score'] += 1 / (k + rank)

        return sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)


    def rerank(query, candidates, top_k=5):
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []

        pairs = [[query, c['text']] for c in candidates]
        scores = reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate['rerank_score'] = float(score)

        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]


    # Define main query function
    def query_librarian(question, return_sources=False):
        """
        Complete RAG pipeline: Hybrid Search → RRF → Rerank → Generate
        """
        # Step 1: Hybrid retrieval
        dense = dense_search(question, top_k=10)
        bm25 = bm25_search(question, top_k=10)

        # Step 2: Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(dense, bm25)

        # Step 3: Cross-encoder reranking
        reranked = rerank(question, fused, top_k=5)

        # Step 4: Build context
        context = "\\n\\n---\\n\\n".join([r['text'] for r in reranked])

        # Step 5: Generate answer
        prompt = f'''Answer based ONLY on context from Uber's 2024 Annual Report.

Context:
{context}

Question: {question}

Answer:'''

        response = llm.invoke(prompt)
        answer = response.content

        if return_sources:
            return {
                'answer': answer,
                'sources': reranked,
                'context': context
            }
        return answer


    print("✅ query_librarian() ready!")

except Exception as e:
    raise e
    # print(f"❌ Error loading RAG system: {e}")
    # print("   Please run Notebook 03 first!")
    #
    #
    # # Placeholder function
    # def query_librarian(question, return_sources=False):
    #     if return_sources:
    #         return {
    #             'answer': "[Error: Run Notebook 03 first]",
    #             'context': ''
    #         }
    #     return "[Error: Run Notebook 03 first]"
    #
    #
    # print("⚠️  Using placeholder query_librarian()")
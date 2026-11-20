"""
Loads both systems for Notebook 04 on ONE base model so the only difference is the method:
- Fine-tuned: base Llama + LoRA adapters, no retrieval
- RAG:        same base Llama with adapters disabled, answering from retrieved context
Both use the same system prompt, decoding settings and token budget.
"""
from pathlib import Path

import torch
import weaviate
from peft import PeftModel
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.services.llm_services import load_config, get_text_embeddings

config = load_config("../src/config/config.yaml")

SYSTEM_PROMPT = "You are a financial analyst trained on Uber's 2024 Annual Report."  # same as Notebook 02
MAX_NEW_TOKENS = 256

# ============================================================
# Shared base model + adapters
# ============================================================

if torch.cuda.is_available():
    device, dtype = "cuda", torch.float16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float16
else:
    device, dtype = "cpu", torch.float32

model_id = config['finetune_model']
adapter_path = Path(config['artifacts_root']) / "adapters" / config['adapter_name']
if not adapter_path.exists():
    raise FileNotFoundError(f"Adapters not found at {adapter_path}, run Notebook 02 and copy them here")

print(f"Loading {model_id} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="eager").to(device)
base_model.generation_config.max_length = None
model = PeftModel.from_pretrained(base_model, str(adapter_path))
model.eval()
print(f"Adapters loaded from {adapter_path}")


def generate(user_content, use_adapter):
    """Greedy decoding, so both systems are deterministic and comparable."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)

    settings = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    with torch.no_grad():
        if use_adapter:
            outputs = model.generate(**inputs, **settings)
        else:
            with model.disable_adapter():
                outputs = model.generate(**inputs, **settings)

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def query_finetuned(question):
    """Fine-tuned model answering from memory."""
    return generate(question, use_adapter=True)


# ============================================================
# RAG: index from Notebook 03
# ============================================================

embeddings = get_text_embeddings(config)

client = weaviate.connect_to_embedded(version="1.27.3")  # same version Notebook 03 indexed with
collection_name = "UberFinancials"
if not client.collections.exists(collection_name):
    raise RuntimeError(f"Collection '{collection_name}' not found, run Notebook 03 first")
collection = client.collections.get(collection_name)

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def dense_search(query, top_k=10):
    response = collection.query.near_vector(
        near_vector=embeddings.embed_query(query), limit=top_k, return_metadata=['distance']
    )
    return [{'chunk_id': o.properties['chunk_id'], 'text': o.properties['text']} for o in response.objects]


def bm25_search(query, top_k=10):
    response = collection.query.bm25(query=query, limit=top_k)
    return [{'chunk_id': o.properties['chunk_id'], 'text': o.properties['text']} for o in response.objects]


def reciprocal_rank_fusion(*result_lists, k=60):
    fused = {}
    for results in result_lists:
        for rank, r in enumerate(results, 1):
            entry = fused.setdefault(r['chunk_id'], {**r, 'rrf_score': 0.0})
            entry['rrf_score'] += 1 / (k + rank)
    return sorted(fused.values(), key=lambda x: x['rrf_score'], reverse=True)


def rerank(query, candidates, top_k=5):
    if not candidates:
        return []
    scores = reranker.predict([[query, c['text']] for c in candidates])
    for c, s in zip(candidates, scores):
        c['rerank_score'] = float(s)
    return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]


def query_rag(question, return_sources=False):
    """Hybrid search -> RRF -> rerank -> base model answers from context."""
    fused = reciprocal_rank_fusion(dense_search(question), bm25_search(question))
    sources = rerank(question, fused, top_k=5)
    context = "\n\n---\n\n".join(s['text'] for s in sources)

    prompt = f"""Answer the question using ONLY the context below from Uber's 2024 Annual Report.

Context:
{context}

Question: {question}"""

    answer = generate(prompt, use_adapter=False)
    if return_sources:
        return {'answer': answer, 'sources': sources, 'context': context}
    return answer


print("query_finetuned() and query_rag() ready")

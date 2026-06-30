import chromadb
import re
import os
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()

groq_client   = Groq()
chroma_client = chromadb.PersistentClient(path="./retail_vector_db")
collection    = chroma_client.get_collection(name="inventory_logs")
embed_model   = SentenceTransformer("all-MiniLM-L6-v2")

N_RESULTS          = 20
DISTANCE_THRESHOLD = 1.0


# ── Log request detector ───────────────────────────────────────────────────────
LOG_KEYWORDS = [
    "work log", "activity log", "what happened", "show me records",
    "show logs", "log entries", "daily log", "raw log", "log for",
    "records for", "entries for", "what was recorded", "show me what happened"
]

def _is_log_request(question: str) -> bool:
    return any(kw in question.lower() for kw in LOG_KEYWORDS)


# ── Retrieval ──────────────────────────────────────────────────────────────────
def _retrieve(query: str, store_id: str = None):
    query_vector = embed_model.encode(query).tolist()
    
    # 1. Extract the date BEFORE querying the database
    date_match  = re.search(r'\d{4}-\d{2}-\d{2}', query)
    month_match = re.search(r'\d{4}-\d{2}(?!-\d{2})', query)  

    # 2. Build ChromaDB 'where' filter dynamically
    where_conditions = []
    
    if store_id:
        where_conditions.append({"store_id": store_id})
        
    if date_match:
        where_conditions.append({"date": date_match.group()})
    # Note: ChromaDB doesn't natively support 'startswith' for strings in standard operators,
    # so for exact month matching, we rely on post-filtering if it's just a month.

    # Combine conditions using Chroma's $and syntax if multiple exist
    if len(where_conditions) == 1:
        where_filter = where_conditions[0]
    elif len(where_conditions) > 1:
        where_filter = {"$and": where_conditions}
    else:
        where_filter = None

    # 3. Query with the hard filter applied
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=N_RESULTS,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    if not docs:
        return [], [], [], False

    # 4. Handle Month post-filtering (since Chroma string matching is exact)
    if not date_match and month_match:
        target = month_match.group()
        filtered = [(d, m, dist) for d, m, dist in zip(docs, metadatas, distances)
                    if m.get("date", "").startswith(target)]
        if filtered:
            docs, metadatas, distances = zip(*filtered)
            docs, metadatas, distances = list(docs), list(metadatas), list(distances)
        else:
            docs, metadatas, distances = [], [], []

    # 5. Distance threshold
    if date_match or month_match or store_id:
        relevant = list(zip(docs, metadatas, distances))
    else:
        # Only apply the strict distance threshold for pure semantic queries
        relevant = [(d, m, dist) for d, m, dist in zip(docs, metadatas, distances)
                    if dist <= DISTANCE_THRESHOLD]

    if not relevant:
        return [], [], [], False

    docs_f, metas_f, dists_f = zip(*relevant)
    return list(docs_f), list(metas_f), list(dists_f), True

# ── Response builders ──────────────────────────────────────────────────────────
def _format_log_output(docs, metadatas) -> str:
    lines = ["📋 **Retrieved Log Entries:**\n"]
    for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
        lines.append(f"**Entry {i}**")
        lines.append(doc)
        lines.append("")
    return "\n".join(lines)


def _build_messages(context: str, question: str, chat_history: list = None) -> list:
    system_prompt = """You are a Retail Intelligence Analyst with access to historical inventory and sales records.

RULES:
1. If the answer is directly in the records, state it confidently with specific numbers.
2. If inference is needed, reason carefully and state your reasoning.
3. If records don't contain enough info, say: "I don't have enough data in the retrieved records to answer this."
4. NEVER use outside knowledge. NEVER write code. Answer in plain English.
5. If a Store ID, Product ID, or date is mentioned but not found, say so explicitly.
6. Keep answers concise. Use bullet points for multiple findings."""

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": f"Retrieved Records:\n{context}\n\nQuestion: {question}"})
    return messages


# ── Main function ──────────────────────────────────────────────────────────────
def generate_rag_response(user_question: str, store_id: str = None, chat_history: list = None) -> str:
    docs, metadatas, distances, found = _retrieve(user_question, store_id=store_id)

    if not found:
        scope = f" for Store {store_id}" if store_id else ""
        return (
            f"No closely matching records were found{scope}. "
            "Try rephrasing with specific store IDs, product IDs, categories, or date ranges."
        )

    # Raw log request → return document strings directly, skip LLM
    if _is_log_request(user_question):
        return _format_log_output(docs, metadatas)

    # Synthesize with LLM
    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
        meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
        context_parts.append(f"[Record {i} | {meta_str}]\n{doc}")

    context  = "\n\n".join(context_parts)
    messages = _build_messages(context, user_question, chat_history)

    try:
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=600,
            timeout=20.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: Could not reach AI service. Detail: {str(e)}"


# ── Direct run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Retail RAG Assistant ---")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        vec = embed_model.encode(user_input).tolist()
        raw = collection.query(query_embeddings=[vec], n_results=8, include=["distances"])
        print(f"[DEBUG distances]: {[round(d, 3) for d in raw['distances'][0]]}")
        print("Thinking...")
        print(f"\nAI: {generate_rag_response(user_input)}\n")
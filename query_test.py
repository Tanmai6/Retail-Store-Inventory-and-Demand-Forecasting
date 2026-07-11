# import chromadb
# import re
# import os
# from groq import Groq
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer

# # ── Setup ──────────────────────────────────────────────────────────────────────
# load_dotenv()

# groq_client   = Groq()
# chroma_client = chromadb.PersistentClient(path="./retail_vector_db")
# collection    = chroma_client.get_collection(name="inventory_logs")
# embed_model   = SentenceTransformer("all-MiniLM-L6-v2")

# N_RESULTS          = 20
# DISTANCE_THRESHOLD = 1.0


# # ── Retrieval ──────────────────────────────────────────────────────────────────
# def _retrieve(query: str, store_id: str = None):
#     query_vector = embed_model.encode(query).tolist()
    
#     # 1. Extract the date BEFORE querying the database
#     date_match  = re.search(r'\d{4}-\d{2}-\d{2}', query)
#     month_match = re.search(r'\d{4}-\d{2}(?!-\d{2})', query)  

#     # 2. Build ChromaDB 'where' filter dynamically
#     where_conditions =  []
#     store_match = re.search(r'\b[Ss]\d{3}\b', query, re.IGNORECASE)
#     effective_store_id = store_id or (store_match.group().upper() if store_match else None)
#     if effective_store_id:
#         where_conditions.append({"store_id": effective_store_id})
        
#     if date_match:
#         where_conditions.append({"date": date_match.group()})

#     # Combine conditions using Chroma's $and syntax if multiple exist
#     if len(where_conditions) == 1:
#         where_filter = where_conditions[0]
#     elif len(where_conditions) > 1:
#         where_filter = {"$and": where_conditions}
#     else:
#         where_filter = None

#     # 3. Query with the hard filter applied
#     results = collection.query(
#         query_embeddings=[query_vector],
#         n_results=N_RESULTS,
#         where=where_filter,
#         include=["documents", "metadatas", "distances"]
#     )

#     docs      = results["documents"][0]
#     metadatas = results["metadatas"][0]
#     distances = results["distances"][0]
    
#     if not docs:
#         return [], [], [], False

#     # 4. Handle Month post-filtering (since Chroma string matching is exact)
#     if not date_match and month_match:
#         target = month_match.group()
#         filtered = [(d, m, dist) for d, m, dist in zip(docs, metadatas, distances)
#                     if m.get("date", "").startswith(target)]
#         if filtered:
#             docs, metadatas, distances = zip(*filtered)
#             docs, metadatas, distances = list(docs), list(metadatas), list(distances)
#         else:
#             docs, metadatas, distances = [], [], []

#     # 5. Distance threshold
#     if date_match or month_match or store_id:
#         relevant = list(zip(docs, metadatas, distances))
#     else:
#         # Only apply the strict distance threshold for pure semantic queries
#         relevant = [(d, m, dist) for d, m, dist in zip(docs, metadatas, distances)
#                     if dist <= DISTANCE_THRESHOLD]

#     if not relevant:
#         return [], [], [], False

#     docs_f, metas_f, dists_f = zip(*relevant)
#     return list(docs_f), list(metas_f), list(dists_f), True

# # ── Response builders ──────────────────────────────────────────────────────────
# def _format_log_output(docs, metadatas) -> str:
#     lines = ["📋 **Retrieved Log Entries:**\n"]
#     for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
#         lines.append(f"**Entry {i}**")
#         lines.append(doc)
#         lines.append("")
#     return "\n".join(lines)


# def _build_messages(context: str, question: str, chat_history: list = None) -> list:
#     system_prompt = """You are a Retail Intelligence Analyst with access to historical inventory and sales records.

# RULES:
# 1. If the answer is directly in the records, state it confidently with specific numbers.
# 2. If inference is needed, reason carefully and state your reasoning.
# 3. If records don't contain enough info, say: "I don't have enough data in the retrieved records to answer this."
# 4. NEVER use outside knowledge. NEVER write code. Answer in plain English.
# 5. If a Store ID, Product ID, or date is mentioned but not found, say so explicitly.
# 6. Keep answers concise. Use bullet points for multiple findings."""

#     messages = [{"role": "system", "content": system_prompt}]
#     if chat_history:
#         messages.extend(chat_history)
#     messages.append({"role": "user", "content": f"Retrieved Records:\n{context}\n\nQuestion: {question}"})
#     return messages


# # ── Main function ──────────────────────────────────────────────────────────────
# def generate_rag_response(user_question: str, store_id: str = None, chat_history: list = None) -> str:
#     docs, metadatas, distances, found = _retrieve(user_question, store_id=store_id)

#     if not found:
#         scope = f" for Store {store_id}" if store_id else ""
#         return (
#             f"No closely matching records were found{scope}. "
#             "Try rephrasing with specific store IDs, product IDs, categories, or date ranges."
#         )

#     # Synthesize with LLM
#     context_parts = []
#     for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
#         meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
#         context_parts.append(f"[Record {i} | {meta_str}]\n{doc}")

#     context  = "\n\n".join(context_parts)
#     messages = _build_messages(context, user_question, chat_history)

#     try:
#         response = groq_client.chat.completions.create(
#             messages=messages,
#             model="llama-3.1-8b-instant",
#             temperature=0.0,
#             max_tokens=600,
#             timeout=20.0
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"Error: Could not reach AI service. Detail: {str(e)}"


# # ── Direct run ─────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("\n--- Retail RAG Assistant ---")
#     while True:
#         user_input = input("You: ").strip()
#         if not user_input:
#             continue
#         if user_input.lower() in ("exit", "quit"):
#             break
#         vec = embed_model.encode(user_input).tolist()
#         raw = collection.query(query_embeddings=[vec], n_results=8, include=["distances"])
#         print(f"[DEBUG distances]: {[round(d, 3) for d in raw['distances'][0]]}")
#         print("Thinking...")
#         print(f"\nAI: {generate_rag_response(user_input)}\n")





"""
Handles unstructured knowledge - policy documents, SOPs, free-text notes,
manuals, or anything NOT stored as rows in Master_View.
Examples: "What is our return policy for damaged goods?",
"Summarize the supplier onboarding guidelines",
"What does the SOP say about handling stockouts?"

This module no longer answers retail-data questions. It only queries the
company_policies collection built by policy_ingest.py.
"""

import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Setup ────────────────────────────────────────────────────────────────────
load_dotenv()

groq_client   = Groq()
chroma_client = chromadb.PersistentClient(path="./retail_vector_db")
collection    = chroma_client.get_collection(name="company_policies")
embed_model   = SentenceTransformer("all-MiniLM-L6-v2")

N_RESULTS           = 5     # cap retrieved chunks to control context size
DISTANCE_THRESHOLD  = 1.0   # above this, we treat a match as not relevant
MAX_CHARS_PER_CHUNK = 1200  # safety truncation per chunk, sections are short so rarely hit
MAX_HISTORY_TURNS   = 6     # keep only the last N messages of chat history
MAX_TOKENS          = 400   # keep completions short and predictable

# Same keyword -> section_number map used at ingestion time, kept here so
# retrieval can apply metadata filtering without re-reading the PDF.
SECTION_KEYWORDS = {
    1:  ["company overview", "overview", "how many stores", "who is the ceo", "ceo"],
    2:  ["organizational structure", "org structure", "manager", "reporting", "hierarchy"],
    3:  ["store operating", "store hours", "opening hours", "safety stock", "stockout"],
    4:  ["warehouse", "inventory policy", "replenishment", "reorder", "seasonal stock"],
    5:  ["pricing", "promotions", "competitor pricing", "discount policy"],
    6:  ["return", "refund", "exchange", "damaged goods", "clearance", "sale item"],
    7:  ["customer service", "complaint", "feedback"],
    8:  ["leave", "attendance", "sick leave", "paid leave", "absence"],
    9:  ["expense", "reimbursement", "claim"],
    10: ["vendor", "procurement", "purchase order", "supplier", "onboarding"],
    11: ["it policy", "data security", "data breach", "access control", "role based"],
    12: ["health and safety", "fire safety", "protective equipment", "workplace incident"],
    13: ["escalation", "escalation matrix", "who do i contact", "who handles"],
}


def _matching_sections(query: str):
    """Return section_numbers whose keywords appear in the query, else []."""
    q = query.lower()
    matches = [num for num, kws in SECTION_KEYWORDS.items() if any(kw in q for kw in kws)]
    return matches


# ── Retrieval ────────────────────────────────────────────────────────────────
def _retrieve(query: str):
    query_vector = embed_model.encode(query).tolist()
    matched_sections = _matching_sections(query)

    candidates = {}  # chunk_id -> (doc, meta, dist)

    def _merge(results):
        ids       = results["ids"][0]
        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metadatas, distances):
            if cid not in candidates or dist < candidates[cid][2]:
                candidates[cid] = (doc, meta, dist)

    # Keyword-filtered search (fast path to the likely-correct section)
    if matched_sections:
        filtered_results = collection.query(
            query_embeddings=[query_vector],
            n_results=N_RESULTS,
            where={"section_number": {"$in": matched_sections}},
            include=["documents", "metadatas", "distances"],
        )
        _merge(filtered_results)

    # Always also run a plain unfiltered semantic search. This prevents a
    # keyword match to the WRONG section (e.g. "warehouse" matching Section 4
    # "Warehouse and Inventory Policy" for a question that's really about
    # Section 13 "Escalation Matrix") from hiding the actually relevant chunk.
    unfiltered_results = collection.query(
        query_embeddings=[query_vector],
        n_results=N_RESULTS,
        include=["documents", "metadatas", "distances"],
    )
    _merge(unfiltered_results)

    if not candidates:
        return [], [], [], False

    ranked = sorted(candidates.values(), key=lambda x: x[2])
    relevant = [(d, m, dist) for d, m, dist in ranked if dist <= DISTANCE_THRESHOLD][:N_RESULTS]

    if not relevant:
        return [], [], [], False

    docs_f, metas_f, dists_f = zip(*relevant)
    return list(docs_f), list(metas_f), list(dists_f), True


# ── Response builders ────────────────────────────────────────────────────────
def _build_messages(context: str, question: str, chat_history: list = None) -> list:
    system_prompt = """You are a Company Policy Assistant with access to RetailCo's internal
policy handbook (returns, HR, vendor, IT, safety, escalation, etc.).

RULES:
1. Answer only using the retrieved policy text below. NEVER use outside knowledge.
2. If the retrieved text directly answers the question, state it clearly, including any
   numbers, thresholds, or timeframes mentioned.
3. If the retrieved text is only partially relevant, say what it does cover and note
   what it does not.
4. If the retrieved text does not answer the question, say:
   "I don't have information about that in the policy documents."
5. Keep answers concise. Use bullet points for multi-part answers."""

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        # Truncate history so context never grows unbounded.
        messages.extend(chat_history[-MAX_HISTORY_TURNS:])

    messages.append({"role": "user", "content": f"Retrieved Policy Sections:\n{context}\n\nQuestion: {question}"})
    return messages


def _format_context(docs, metadatas) -> str:
    parts = []
    for doc, meta in zip(docs, metadatas):
        text = doc[:MAX_CHARS_PER_CHUNK]
        parts.append(f"[Section {meta['section_number']}: {meta['section_title']}]\n{text}")
    return "\n\n".join(parts)


# ── Main function ────────────────────────────────────────────────────────────
def generate_rag_response(user_question: str, store_id: str = None, chat_history: list = None) -> str:
    """
    store_id is accepted for interface compatibility with the orchestrator
    (router.py calls generate_rag_response(query, store_id=store_id) for
    both SQL and RAG). Policy content is company-wide and not scoped by
    store, so this parameter is intentionally unused here.
    """
    docs, metadatas, distances, found = _retrieve(user_question)

    if not found:
        return (
            "I don't have information about that in the policy documents. "
            "Try rephrasing, or ask about a specific policy area (returns, leave, "
            "vendor onboarding, IT security, escalation, etc.)."
        )

    context  = _format_context(docs, metadatas)
    messages = _build_messages(context, user_question, chat_history)

    try:
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            timeout=20.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Catches context-length / rate-limit / API errors alike so a token
        # limit issue never surfaces as a crash -- always return a message.
        err_str = str(e).lower()
        if "context" in err_str or "token" in err_str or "length" in err_str:
            return (
                "The retrieved policy content was too long to process in one go. "
                "Try asking a more specific question about a single policy area."
            )
        return f"Error: Could not reach AI service. Detail: {str(e)}"


# ── Direct run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Company Policy RAG Assistant ---")
    history = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        vec = embed_model.encode(user_input).tolist()
        raw = collection.query(query_embeddings=[vec], n_results=N_RESULTS, include=["distances"])
        print(f"[DEBUG distances]: {[round(d, 3) for d in raw['distances'][0]]}")
        print("Thinking...")

        answer = generate_rag_response(user_input, chat_history=history)
        print(f"\nAI: {answer}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        history = history[-MAX_HISTORY_TURNS:]

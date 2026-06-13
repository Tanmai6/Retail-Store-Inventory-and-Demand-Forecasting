# import chromadb
# import os
# from groq import Groq
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer

# # 1. Setup
# load_dotenv() # Ensures it finds the .env file in root
# groq_client = Groq()
# # Pointing to the database created by main.py
# chroma_Client = chromadb.PersistentClient(path='./retail_vector_db')
# collection = chroma_Client.get_collection(name="inventory_logs")
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # ... (Keep all your imports and setup at the top exactly as they are) ...

# # 1. Update the function to take just the query (to match the orchestrator)
# def generate_rag_response(user_question):
#     # 1. RETRIEVE
#     query_vector = model.encode(user_question).tolist()
#     result = collection.query(query_embeddings=[query_vector], n_results=3)
   
#     retrieved_context = "\n".join(result['documents'][0])
    
#     # 2. SYSTEM PROMPT
#     system_prompt_content = (
#         "You are a Retail Intelligence Analyst. "
#         "Extract the information directly from the 'Historical Context' provided. "
#         "If you cannot find the answer, say 'I cannot find that in the provided logs'."
#     )
    
#     # 3. CONTEXTUAL PROMPT
#     user_prompt = f"Historical Context:\n{retrieved_context}\n\nUser Question: {user_question}"
    
#     # 4. BUILD PAYLOAD
#     messages_payload = [
#         {"role": "system", "content": system_prompt_content},
#         {"role": "user", "content": user_prompt}
#     ]
    
#     # 5. SYNTHESIZE with Timeout
#     try:
#         chat_completion = groq_client.chat.completions.create(
#             messages=messages_payload,
#             model="llama-3.1-8b-instant",
#             temperature=0.0,
#             timeout=15.0 
#         )
#         return chat_completion.choices[0].message.content
#     except Exception as e:
#         return f"Error: Could not connect to AI service. {str(e)}"


# # 2. WRAP THE RUNLOOP
# # This ensures it only runs if you execute query_test.py directly, 
# # but stays quiet when orchestrator.py imports it.
# if __name__ == "__main__":
#     print("\n--- Retail AI Assistant Operational ---")
    
#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() in ['exit', 'quit']: break
        
#         print("Thinking...")
#         ai_answer = generate_rag_response(user_input)
#         print(f"\nAI: {ai_answer}")



import chromadb
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

# Tunable — run debug mode first to calibrate distance values
N_RESULTS          = 8
DISTANCE_THRESHOLD = 1.0   # cosine distance: 0=identical, 2=opposite; tune after rebuild


def _retrieve_context(query: str) -> tuple[str, bool]:
    query_vector = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=N_RESULTS,
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    relevant = [
        (doc, meta, dist)
        for doc, meta, dist in zip(docs, metadatas, distances)
        if dist <= DISTANCE_THRESHOLD
    ]

    if not relevant:
        return "", False

    context_parts = []
    for i, (doc, meta, dist) in enumerate(relevant, 1):
        meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
        context_parts.append(f"[Record {i} | {meta_str}]\n{doc}")

    return "\n\n".join(context_parts), True


def _build_messages(context: str, question: str, chat_history: list = None) -> list:
    system_prompt = """You are a Retail Intelligence Analyst with access to historical inventory and sales records.

RULES (follow strictly in order):
1. If the answer is directly present or clearly inferable from the records below, state it confidently. Note: records cover historical data so 'current status' means the most recent records available.
2. If the answer requires reasonable inference FROM the records (trends, comparisons, aggregations), reason carefully and state your reasoning explicitly.
3. If the records genuinely do not contain enough information — even with inference — respond with exactly: "I don't have enough data in the retrieved records to answer this."
4. NEVER use knowledge from outside the provided records.
5. NEVER write code. Answer in plain English with specific numbers and facts from the records.
6. If a Store ID, Product ID, or date is mentioned in the question but not in the records, say it was not found in the retrieved sample."""

    user_prompt = f"""Retrieved Records:
{context}

Question: {question}"""

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_rag_response(user_question: str, chat_history: list = None) -> str:
    context, found = _retrieve_context(user_question)

    if not found:
        return (
            "No closely matching records were found in the database for your question. "
            "Try rephrasing with specific store IDs, categories, or date ranges."
        )

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
    print(f"Threshold: {DISTANCE_THRESHOLD} | Top-N: {N_RESULTS}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        # Debug: show distances so you can tune DISTANCE_THRESHOLD
        vec = embed_model.encode(user_input).tolist()
        raw = collection.query(
            query_embeddings=[vec],
            n_results=N_RESULTS,
            include=["distances"]
        )
        print(f"[DEBUG distances]: {[round(d, 3) for d in raw['distances'][0]]}")

        print("Thinking...")
        print(f"\nAI: {generate_rag_response(user_input)}\n")
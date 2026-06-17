import chromadb
import os
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. Setup
load_dotenv() # Ensures it finds the .env file in root
groq_client = Groq()
# Pointing to the database created by main.py
chroma_Client = chromadb.PersistentClient(path='./retail_vector_db')
collection = chroma_Client.get_collection(name="inventory_logs")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ... (Keep all your imports and setup at the top exactly as they are) ...

# 1. Update the function to take just the query (to match the orchestrator)
def generate_rag_response(user_question, store_id: str = None):
    query_vector = model.encode(user_question).tolist()

    # Build ChromaDB where filter if store is scoped
    where_filter = {"store_id": store_id} if store_id else None

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        where=where_filter  # None = no filter = all stores
    )

    retrieved_context = "\n".join(result['documents'][0])

    store_context = f"You are analysing data ONLY for Store {store_id}." if store_id else \
                    "You have access to data across all stores."

    system_prompt_content = (
        f"You are a Retail Intelligence Analyst. {store_context} "
        "Extract the information directly from the 'Historical Context' provided. "
        "If you cannot find the answer, say 'I cannot find that in the provided logs'."
    )

    user_prompt = f"Historical Context:\n{retrieved_context}\n\nUser Question: {user_question}"
    # ... rest unchanged

# 2. WRAP THE RUNLOOP
# This ensures it only runs if you execute query_test.py directly, 
# but stays quiet when orchestrator.py imports it.
if __name__ == "__main__":
    print("\n--- Retail AI Assistant Operational ---")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        print("Thinking...")
        ai_answer = generate_rag_response(user_input)
        print(f"\nAI: {ai_answer}")
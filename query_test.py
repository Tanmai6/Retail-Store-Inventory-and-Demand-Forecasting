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
def generate_rag_response(user_question):
    # 1. RETRIEVE
    query_vector = model.encode(user_question).tolist()
    result = collection.query(query_embeddings=[query_vector], n_results=3)
   
    retrieved_context = "\n".join(result['documents'][0])
    
    # 2. SYSTEM PROMPT
    system_prompt_content = (
        "You are a Retail Intelligence Analyst. "
        "Extract the information directly from the 'Historical Context' provided. "
        "If you cannot find the answer, say 'I cannot find that in the provided logs'."
    )
    
    # 3. CONTEXTUAL PROMPT
    user_prompt = f"Historical Context:\n{retrieved_context}\n\nUser Question: {user_question}"
    
    # 4. BUILD PAYLOAD
    messages_payload = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_prompt}
    ]
    
    # 5. SYNTHESIZE with Timeout
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages_payload,
            model="llama-3.1-8b-instant",
            temperature=0.0,
            timeout=15.0 
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: Could not connect to AI service. {str(e)}"


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
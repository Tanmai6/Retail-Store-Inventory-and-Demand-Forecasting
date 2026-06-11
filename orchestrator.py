import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from query_test import generate_rag_response
# Import your existing executors
from sql_agent import agent_executor  # Make sure sql_agent.py exposes agent_executor
# from rag_pipeline import run_rag_query  # Uncomment when your RAG file is ready

load_dotenv()

# 1. Router Setup
router_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
router_prompt = PromptTemplate.from_template("""
You are a routing assistant for a retail company. 
Your job is to read the user's query and decide which system should handle it.

Rules:
1. If the user asks about numbers, metrics, inventory, stockouts, revenue, store performance, or data calculations, output EXACTLY the word: SQL
2. If the user asks about company policies, return guidelines, product manuals, or text-based knowledge, output EXACTLY the word: RAG

Do not output any other text, explanation, or punctuation. Just the single word.

User Query: {query}
""")
router_chain = router_prompt | router_llm | StrOutputParser()

# 2. Main Entry Function
def handle_query(query: str):
    # Route the query
    decision = router_chain.invoke({"query": query}).strip().upper()
    
    if "SQL" in decision:
        response = agent_executor.invoke({"input": query})
        return response["output"]
    elif "RAG" in decision:
        # response = run_rag_query(query) # Uncomment when ready
        return "RAG pipeline response placeholder"
    else:
        return "I am not sure how to route this question. Please ask about retail metrics or store policies."

# 3. Execution Block (Fixed function names here)
if __name__ == "__main__":
    # Test 1: Should trigger SQL routing
    print(handle_query("Which store had the most stockouts last month?"))

    # Test 2: Should trigger RAG routing
    print(handle_query("What is the standard operating procedure for handling a broken product?"))
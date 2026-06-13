# import os
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_groq import ChatGroq
# from query_test import generate_rag_response
# # Import your existing executors
# from sql_agent import agent_executor  # Make sure sql_agent.py exposes agent_executor
# # from rag_pipeline import run_rag_query  # Uncomment when your RAG file is ready

# load_dotenv()

# # 1. Router Setup
# router_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
# router_prompt = PromptTemplate.from_template("""
# You are a routing assistant for a retail company. 
# Your job is to read the user's query and decide which system should handle it.

# Rules:
# 1. If the user asks about numbers, metrics, inventory, stockouts, revenue, store performance, or data calculations, output EXACTLY the word: SQL
# 2. If the user asks about company policies, return guidelines, product manuals, or text-based knowledge, output EXACTLY the word: RAG

# Do not output any other text, explanation, or punctuation. Just the single word.

# User Query: {query}
# """)
# router_chain = router_prompt | router_llm | StrOutputParser()

# # 2. Main Entry Function
# def handle_query(query: str):
#     # Route the query
#     decision = router_chain.invoke({"query": query}).strip().upper()
    
#     if "SQL" in decision:
#         response = agent_executor.invoke({"input": query})
#         return response["output"]
#     elif "RAG" in decision:
#         # response = run_rag_query(query) # Uncomment when ready
#         return generate_rag_response(query)
#     else:
#         return "I am not sure how to route this question. Please ask about retail metrics or store policies."

# # 3. Execution Block (Fixed function names here)
# if __name__ == "__main__":
#     # Test 1: Should trigger SQL routing
#     print(handle_query("Which store had the most stockouts last month?"))

#     # Test 2: Should trigger RAG routing
#     print(handle_query("What is the standard operating procedure for handling a broken product?"))






import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from query_test import generate_rag_response
from sql_agent import agent_executor

load_dotenv()

# ── Router ─────────────────────────────────────────────────────────────────────
router_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

router_prompt = PromptTemplate.from_template("""
You are a query router for a retail analytics system. 
Read the user's question and decide which system should handle it.

System A — SQL:
Handles precise aggregations, counts, sums, comparisons, rankings, and filtering.
Examples: "Which store had highest revenue?", "How many stockouts in January?", 
"Compare sales across regions", "Top 5 products by units sold"

System B — RAG:
Handles qualitative questions, pattern exploration, log-level details, 
contextual summaries, and questions about specific records or events.
Examples: "Show me stockout logs for Electronics", "What happened at Store S001 last week?",
"Find records where inventory was critically low", "Describe the sales pattern for Clothing"

Output ONLY one word: SQL or RAG
No explanation, no punctuation, just the single word.

User Query: {query}
""")

router_chain = router_prompt | router_llm | StrOutputParser()


# ── Main Handler ───────────────────────────────────────────────────────────────
def handle_query(query: str, chat_history: list = None) -> str:
    try:
        decision = router_chain.invoke({"query": query}).strip().upper()
    except Exception as e:
        return f"Routing error: {str(e)}"

    if "SQL" in decision:
        try:
            response = agent_executor.invoke({"input": query})
            return response.get("output", "The SQL agent did not return a result.")
        except Exception as e:
            return f"SQL agent error: {str(e)}"

    elif "RAG" in decision:
        try:
            return generate_rag_response(query, chat_history)
        except Exception as e:
            return f"RAG pipeline error: {str(e)}"

    else:
        # Fallback: try RAG since it's more forgiving than SQL
        try:
            return generate_rag_response(query, chat_history)
        except Exception as e:
            return f"Could not process query. Router returned: '{decision}'. Error: {str(e)}"


# ── Direct run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "Which store had the most stockouts last month?",           # → SQL
        "Show me log entries where Electronics items ran out",      # → RAG
        "What is the total revenue across all stores in January?",  # → SQL
        "Find records where we lost demand due to low inventory",   # → RAG
        "What is the recommended temperature for storing groceries?" # → RAG (correct: I don't know)
    ]
    for q in tests:
        print(f"\nQ: {q}")
        print(f"A: {handle_query(q)}")
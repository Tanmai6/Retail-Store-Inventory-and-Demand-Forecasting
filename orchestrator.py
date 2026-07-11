import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from query_test import generate_rag_response
from sql_agent import create_sql_agent_for_scope

load_dotenv()

router_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
 
router_prompt = PromptTemplate.from_template("""
You are a query router for a retail analytics system.
Read the user's question and decide which system should handle it.
 
System A - SQL:
Handles numbers, metrics, inventory, stockouts, revenue, store performance,
calculations, comparisons, rankings, trends, promotion impact, AND any
request for specific records, logs, or "what happened" events - these are
all structured rows in Master_View and should be queried/summarized via SQL.
If the question asks about the IMPACT, EFFECT, INFLUENCE, or RELATIONSHIP of ANY factor
on sales, revenue, inventory, stockouts, or demand - route to SQL, ALWAYS.
This includes: epidemic, weather, promotions, seasonality, discounts, region,
category, store - regardless of how the question is phrased.                                           
Examples: "Which store had highest revenue?", "How many stockouts in January?",
"Compare sales across regions", "Top 5 products by units sold",
"What is the stockout risk?", "How do promotions impact sales?",
"Which region is underperforming?", "Which category earns most?",
"Show me stockout logs for Electronics", "What happened at Store S001 last week?",
"Show me the work log for today","Which region is underperforming?", "Which category earns most?", "What was recorded for Product P001?",
"Find records where inventory was critically low"
 
System B - RAG:
Handles unstructured knowledge - policy documents, SOPs, free-text notes,
manuals, or anything NOT stored as rows in Master_View.
Examples: "What is our return policy for damaged goods?",
"Summarize the supplier onboarding guidelines",
"What does the SOP say about handling stockouts?",
"Who do I escalate a warehouse issue to?",
"What is the escalation ladder for store-level complaints?",
"Who is the CEO of the company?",
"How many stores does the company have?",
"What is the organizational structure?"
                                             
System C - NR
whenever the question is totally unrelated to anything in System A or System B
Examples: "What is capital of India?","Who is Mahatma Gandhi?" 
 
Output ONLY one word: SQL or RAG or NR. No explanation, no punctuation.
 
User Query: {query}
""")
router_chain = router_prompt | router_llm | StrOutputParser()


def handle_query(query: str, store_id: str = None):
    """
    Routes query to SQL agent or RAG pipeline.
    store_id: if set, constrains both pipelines to that store only.
              if None (CEO / Warehouse), all store data is accessible.
    """
    decision = router_chain.invoke({"query": query}).strip().upper()

    if "SQL" in decision:
        agent = create_sql_agent_for_scope(store_id=store_id)
        response = agent.invoke({"input": query})
        return response["output"]

    elif "RAG" in decision:
        return generate_rag_response(query, store_id=store_id)

    else:
        return "I'm not sure about this question. Please ask about retail metrics or store policies."
    

# ──────────────────────────────────────────────
# MANUAL TEST SUITE
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Manual Router Tests...\n")
    print("=" * 70)
    
    # Format: (store_id, query, expected_routing)
    test_cases = [
        # --- SQL Routing Tests (Metrics, Logs, Impacts) ---
        (None,   "Which is the capital of India?", "SQL")
        
    ]

    for i, (store_id, query, expected) in enumerate(test_cases, 1):
        scope_label = f"Store {store_id}" if store_id else "All Stores (Global)"
        
        print(f"Test {i}: {query}")
        print(f"Scope    : {scope_label}")
        print(f"Expected : {expected}")
        
        try:
            # 1. Test just the router decision first to ensure accuracy
            decision = router_chain.invoke({"query": query}).strip().upper()
            
            # Simple color coding for terminal (Optional but helpful)
            match_status = "✅ PASS" if expected in decision else "❌ FAIL"
            print(f"Actual   : {decision} {match_status}")
            
            # 2. To test the full execution, uncomment the lines below:
            # print("Executing full pipeline...")
            # result = handle_query(query, store_id)
            # print(f"Answer   : {result[:250]}...") # Truncated for readability
            
        except Exception as e:
            print(f"ERROR: {e}")
            
        print("-" * 70)

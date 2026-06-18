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
 
System A — SQL:
Handles numbers, metrics, inventory, stockouts, revenue, store performance,
calculations, comparisons, rankings, trends, and promotion impact.
Examples: "Which store had highest revenue?", "How many stockouts in January?",
"Compare sales across regions", "Top 5 products by units sold",
"What is the stockout risk?", "How do promotions impact sales?",
"Which region is underperforming?", "Which category earns most?"
 
System B — RAG:
Handles log-level details, work logs, and questions about specific records or events.
Examples: "Show me stockout logs for Electronics", "What happened at Store S001 last week?",
"Show me the work log for today", "What was recorded for Product P001?",
"Find records where inventory was critically low"
 
Output ONLY one word: SQL or RAG. No explanation, no punctuation.
 
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
        return "I'm not sure how to route this question. Please ask about retail metrics or store policies."
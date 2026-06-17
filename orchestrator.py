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
You are a routing assistant for a retail company.
Your job is to read the user's query and decide which system should handle it.

Rules:
1. If the user asks about numbers, metrics, inventory, stockouts, revenue, store performance, or data calculations, output EXACTLY the word: SQL
2. If the user asks about company policies, return guidelines, product manuals, or text-based knowledge, output EXACTLY the word: RAG

Do not output any other text, explanation, or punctuation. Just the single word.

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
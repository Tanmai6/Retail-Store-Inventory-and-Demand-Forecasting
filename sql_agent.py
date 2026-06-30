import os
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from dotenv import load_dotenv

load_dotenv()

db_path = "retail_analytics.db"

db = SQLDatabase.from_uri(
    f"sqlite:///{db_path}",
    include_tables=["Master_View"],
    sample_rows_in_table_info=0
)

def create_sql_agent_for_scope(store_id: str = None):
    """
    Returns an agent executor scoped to a single store, or all stores if store_id is None.
    """
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)  
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    if store_id:
        scope_rule = f"""
    === STORE SCOPE ===
    You are assisting the Branch Manager of Store '{store_id}' ONLY.
    EVERY query you write MUST include the filter: WHERE store_id = '{store_id}'
    Do NOT return or reference data from any other store. This is a strict data privacy rule.
    """
    else:
        scope_rule = """
    === STORE SCOPE ===
    You have access to ALL stores. Do not apply any store_id filter unless the user specifically asks about one store.
    """

    prefix = f"""
    You are a retail analytics assistant.
    You have access to ONE table: 'Master_View'.
    This table already contains all data, including revenue, region, category, and promotion status.
    {scope_rule}
    === STRICT RULES ===
    1. NEVER perform a JOIN. All columns are already present in 'Master_View'.
    2. Use the 'promotion' column for inquiries about "Ps" or "promotions".
    3. Use 'AVG(is_stockout) * 100' to calculate "stockout risk".
    4. You must strictly follow the Thought -> Action -> Observation loop.
    5. CRITICAL: ONLY output the Action. Wait for the Observation before giving the Final Answer.
    6. "most revenue", "highest revenue", "earns most" → SUM(revenue)
    === CRITICAL STOP RULE ===
    If you have already found the answer, DO NOT perform any further actions.
    Output the Final Answer immediately. Do not confirm with additional queries.
    === LOOP PREVENTION ===
    Once you have executed a query and received data in an Observation, you MUST write your Final Answer immediately.
    Do NOT re-run the same query. Do NOT run a follow-up query to "confirm" results.
    The pattern is: ONE query → ONE observation → Final Answer. Never more than that for simple aggregations.

    If you already have the data to answer the question, the ONLY valid next step is:
    Thought: I now have the data I need.
    Final Answer: [your answer here]
    """

    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="zero-shot-react-description",
        handle_parsing_errors=True,
        prefix=prefix,
    )
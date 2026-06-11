import os
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from dotenv import load_dotenv

load_dotenv()

# Use hardcoded path to ensure we point to the same DB file as your fix_db.py
db_path = "retail_analytics.db"

# 1. Connect ONLY to the Master_View table
db = SQLDatabase.from_uri(
    f"sqlite:///{db_path}",
    include_tables=["Master_View"], 
    sample_rows_in_table_info=0
)

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# 2. Define the Agent Executor with a simplified prefix
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="zero-shot-react-description",
    handle_parsing_errors=True,          
    early_stopping_method="force",
    prefix="""
    You are a retail analytics assistant. 
    You have access to ONE table: 'Master_View'. 
    This table already contains all data, including revenue, region, category, and promotion status.

    === STRICT RULES ===
    1. NEVER perform a JOIN. All columns are already present in 'Master_View'.
    2. Use the 'promotion' column for inquiries about "Ps" or "promotions".
    3. Use 'AVG(is_stockout) * 100' to calculate "stockout risk".
    4. You must strictly follow the Thought -> Action -> Observation loop.
    5. CRITICAL: ONLY output the Action. Wait for the Observation before giving the Final Answer.
    === CRITICAL STOP RULE ===
    If you have already found the answer to the user's question, DO NOT perform any further actions. 
    You must output the Final Answer immediately after finding the result. 
    Do not try to "confirm" your answer with additional queries.
    """
)
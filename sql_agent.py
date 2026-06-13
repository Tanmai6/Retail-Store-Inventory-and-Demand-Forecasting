# import os
# from langchain_community.agent_toolkits import create_sql_agent
# from langchain_community.utilities import SQLDatabase
# from langchain_groq import ChatGroq
# from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from dotenv import load_dotenv

# load_dotenv()

# # # Use hardcoded path to ensure we point to the same DB file as your fix_db.py
# # db_path = "retail_analytics.db"
# # 1. Lock onto the main project directory
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # 2. Build the exact path to the database
# db_path = os.path.join(BASE_DIR, "retail_analytics.db")

# # 1. Connect ONLY to the Master_View table
# db = SQLDatabase.from_uri(
#     f"sqlite:///{db_path}",
#     include_tables=["Master_View"], 
#     sample_rows_in_table_info=0
# )

# llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# # 2. Define the Agent Executor with a simplified prefix
# agent_executor = create_sql_agent(
#     llm=llm,
#     toolkit=toolkit,
#     verbose=True,
#     agent_type="zero-shot-react-description",
#     handle_parsing_errors=True,          
#     early_stopping_method="force",
#     prefix="""
#     You are a retail analytics assistant. 
#     You have access to ONE table: 'Master_View'. 
#     This table already contains all data, including revenue, region, category, and promotion status.

#     === STRICT RULES ===
#     1. NEVER perform a JOIN. All columns are already present in 'Master_View'.
#     2. Use the 'promotion' column for inquiries about "Ps" or "promotions".
#     3. Use 'AVG(is_stockout) * 100' to calculate "stockout risk".
#     4. You must strictly follow the Thought -> Action -> Observation loop.
#     5. CRITICAL: ONLY output the Action. Wait for the Observation before giving the Final Answer.
#     === CRITICAL STOP RULE ===
#     If you have already found the answer to the user's question, DO NOT perform any further actions. 
#     You must output the Final Answer immediately after finding the result. 
#     Do not try to "confirm" your answer with additional queries.
#     """
# )




import os
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path  = os.path.join(BASE_DIR, "retail_analytics.db")

db = SQLDatabase.from_uri(
    f"sqlite:///{db_path}",
    include_tables=["Master_View"],
    sample_rows_in_table_info=3    # FIX: gives agent real examples of data format
)

llm     = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="zero-shot-react-description",
    handle_parsing_errors=True,
    max_iterations=6,              # FIX: explicit cap instead of "force" stopping
    prefix="""
You are a retail analytics SQL assistant.
You have access to ONE table: 'Master_View'.
This table contains all columns — never perform a JOIN.

Available columns: record_date, store_id, product_id, inventory_level, units_sold, 
demand, price, discount, promotion, revenue, is_stockout, region, location_type, category.

RULES:
1. Always inspect the table schema first with the schema tool before writing SQL.
2. Use AVG(is_stockout) * 100 to calculate stockout rate as a percentage.
3. For promotions, filter WHERE promotion = 1.
4. Date format is YYYY-MM-DD (e.g., '2023-01-15').
5. After getting a query result, immediately provide the Final Answer. Do not run additional queries to verify.
6. If a query returns no rows, say so clearly — do not guess.
"""
)
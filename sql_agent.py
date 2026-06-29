import os
import re
import sqlite3
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DB_PATH = "retail_analytics.db"

# ──────────────────────────────────────────────
# 1. SCHEMA INTROSPECTION  (run once on import)
# ──────────────────────────────────────────────
def _get_master_view_schema() -> str:
    """Pull exact column names + types from the DB at startup.
    Eliminates all column-name hallucination."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Master_View);")
    rows = cursor.fetchall()
    conn.close()
    cols = "\n".join(f"  {r[1]:30s} {r[2]}" for r in rows)
    return f"Master_View columns:\n{cols}"

SCHEMA = _get_master_view_schema()

# ──────────────────────────────────────────────
# 2. SYSTEM PROMPT FACTORY
# ──────────────────────────────────────────────
def _build_system_prompt(store_id: str | None) -> str:
    if store_id:
        scope_rule = (
            f"You are assisting the Branch Manager of Store '{store_id}' ONLY.\n"
            f"EVERY SQL query MUST include: WHERE store_id = '{store_id}'\n"
            "Do NOT return or reference data from any other store."
        )
    else:
        scope_rule = (
            "You have access to ALL stores. Do NOT apply a store_id filter "
            "unless the user explicitly asks about a specific store."
        )

    return f"""You are a retail analytics SQL expert. Your ONLY job is to write ONE valid SQLite query against Master_View and return the result.

{SCHEMA}

=== KEY CONSTRAINTS ===
- product_id is NOT globally unique. It is scoped to a store.
  ALWAYS group/filter by BOTH store_id AND product_id together when doing product-level analysis.
  Example (correct):   GROUP BY store_id, product_id
  Example (WRONG):     GROUP BY product_id           ← mixes products across stores

- NEVER JOIN any tables. All columns are already in Master_View.
- ONLY query Master_View.
- Use ONLY the exact column names listed in the schema above.
- Dates are stored as 'YYYY-MM-DD' text strings.

=== COLUMN QUICK-REFERENCE ===
- Revenue             → revenue  (already computed: units_sold * price * (1 - discount/100))
- Stockout risk       → AVG(is_stockout) * 100
- Promotions          → use 1 (active) or 0 (not active)
- Inventory right now → inventory_level (end-of-day snapshot for that record_date)
- Lost sales          → lost_demand
- Excess stock        → overstock (flag), coverage_days (days of stock on hand)
=== COLUMN VALUE REFERENCE (use EXACTLY these) ===
- seasonality:       'Winter', 'Spring', 'Summer', 'Autumn'
- weather_condition: 'Snowy', 'Rainy', 'Sunny'  
- category:          'Electronics', 'Clothing', 'Groceries', 'Toys', 'Furniture'
- location_type:     'Retail_Branch', 'Warehouse'
- epidemic: INTEGER — use 1 (active) or 0 (not active). NEVER use 'true', 'yes', 'epidemic', or any string.
- promotion:         INTEGER — 1 = promotion active, 0 = no promotion (never use 'true'/'yes')
- is_stockout:       INTEGER — 1 = stockout, 0 = no stockout

=== STORE SCOPE ===
{scope_rule}
=== BUSINESS LOGIC RULES ===
compare regions(north south east west) by their total revenue
=== OUTPUT FORMAT (STRICT) ===
You must reply in this exact format and nothing else:

SQL:
<your single SQL query here>

ANSWER:
<your plain-English answer based on the query result that will be provided to you>

Do NOT explain your reasoning. Do NOT add markdown. Write ONE query only."""


# ──────────────────────────────────────────────
# 3. SQL EXECUTION
# ──────────────────────────────────────────────
def _run_sql(query: str) -> str:
    """Execute a SELECT query and return results as a plain-text table."""
    # Safety: only allow SELECT statements
    stripped = query.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
        return "ERROR: Only SELECT queries are permitted."
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No results found."

        headers = list(rows[0].keys())
        lines = [" | ".join(headers)]
        lines.append("-" * len(lines[0]))
        for row in rows[:100]:     # cap at 100 rows to keep context size sane
            lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))

        if len(rows) > 100:
            lines.append(f"... ({len(rows) - 100} more rows truncated)")

        return "\n".join(lines)

    except sqlite3.Error as e:
        return f"SQL ERROR: {e}"


# ──────────────────────────────────────────────
# 4. SQL EXTRACTION FROM LLM OUTPUT
# ──────────────────────────────────────────────
def _extract_sql(text: str) -> str | None:
    """Pull the SQL block out of the LLM's structured response."""
    # Try the strict format first
    m = re.search(r"SQL:\s*\n(.*?)(?:\nANSWER:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: grab anything that looks like a SELECT
    m = re.search(r"(SELECT\b.*?;)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_answer(text: str) -> str:
    """Pull the ANSWER block, or return the full text as fallback."""
    m = re.search(r"ANSWER:\s*\n(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


# ──────────────────────────────────────────────
# 5. IN-SESSION CACHE
# ──────────────────────────────────────────────
_cache: dict[str, str] = {}

def _cache_key(store_id: str | None, question: str) -> str:
    scope = store_id or "ALL"
    return f"{scope}::{question.strip().lower()}"


# ──────────────────────────────────────────────
# 6. DIRECT SQL CHAIN  (the core logic)
# ──────────────────────────────────────────────
def run_sql_agent(question: str, store_id: str | None = None) -> dict:
    """
    Ask a natural-language question about retail data.

    Returns:
        {
            "answer":    str,   # plain-English response
            "sql":       str,   # the query that was executed
            "raw_data":  str,   # raw table output from SQLite
            "from_cache": bool
        }
    """
    key = _cache_key(store_id, question)
    if key in _cache:
        return {**_cache[key], "from_cache": True}

    client = Groq()
    system_prompt = _build_system_prompt(store_id)

    # ── Step 1: Generate SQL ──────────────────
    sql_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": f"Question: {question}\n\nWrite the SQL query only. Do not execute it yet."},
        ],
    )
    llm_sql_output = sql_response.choices[0].message.content
    sql_query = _extract_sql(llm_sql_output)

    if not sql_query:
        return {
            "answer":    "I could not generate a valid SQL query for that question.",
            "sql":       "",
            "raw_data":  "",
            "from_cache": False,
        }

    # ── Step 2: Execute SQL ───────────────────
    raw_data = _run_sql(sql_query)

    # ── Step 3: Format answer in natural language ─
    answer_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a retail analytics assistant. "
                    "Given a question and SQL query results, write a clear, concise answer. "
                    "Do NOT mention SQL, tables, or columns. Speak to a business user. "
                    "If results are empty, say no data was found. "
                    "Be specific: include numbers, store IDs, and category names from the results."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"SQL executed:\n{sql_query}\n\n"
                    f"Query results:\n{raw_data}\n\n"
                    "Write the business answer:"
                ),
            },
        ],
    )
    answer = answer_response.choices[0].message.content.strip()

    result = {
        "answer":    answer,
        "sql":       sql_query,
        "raw_data":  raw_data,
        "from_cache": False,
    }
    _cache[key] = {k: v for k, v in result.items() if k != "from_cache"}
    return result


# ──────────────────────────────────────────────
# 7. FACTORY FUNCTION  (matches your existing API surface)
# ──────────────────────────────────────────────
class DirectSQLChain:
    """Drop-in replacement for the LangChain agent executor."""

    def __init__(self, store_id: str | None = None):
        self.store_id = store_id

    def invoke(self, inputs: dict) -> dict:
        question = inputs.get("input", "")
        result = run_sql_agent(question, self.store_id)
        return {"output": result["answer"], "sql": result["sql"], "raw_data": result["raw_data"]}

    def run(self, question: str) -> str:
        """Convenience method for simple string-in / string-out usage."""
        return run_sql_agent(question, self.store_id)["answer"]


def create_sql_agent_for_scope(store_id: str | None = None) -> DirectSQLChain:
    """
    Returns a DirectSQLChain scoped to a single store, or all stores if store_id is None.
    API-compatible with the old LangChain agent factory.
    """
    return DirectSQLChain(store_id=store_id)


# ──────────────────────────────────────────────
# 8. MANUAL TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        (None,   "Which store had the highest total revenue?"),
        ("S001", "What are my top 3 products by revenue?"),
        ("S001", "What is my current stockout risk by category?"),
        (None,   "Which region has the most overstock?"),
    ]

    for store_id, question in test_cases:
        scope_label = f"Store {store_id}" if store_id else "All Stores"
        print(f"\n{'='*60}")
        print(f"[{scope_label}] {question}")
        print('='*60)
        result = run_sql_agent(question, store_id)
        print(f"SQL      : {result['sql']}")
        print(f"Raw Data : {result['raw_data'][:300]}...")
        print(f"Answer   : {result['answer']}")
        print(f"Cached   : {result['from_cache']}")
import sqlite3
import os

# Get the absolute path right next to this script
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retail_analytics.db")
print(f"Checking database at: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view');")
    items = cursor.fetchall()
    
    print("\n=== ITEMS FOUND ===")
    if not items:
        print("DATABASE IS COMPLETELY EMPTY!")
    for item in items:
        print(f"- {item[0]}")
    print("===================\n")
except Exception as e:
    print(f"Error reading database: {e}")
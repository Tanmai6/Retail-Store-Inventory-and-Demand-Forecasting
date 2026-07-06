import sqlite3, os

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retail_analytics.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("PRAGMA table_info(Master_View);")
for row in cur.fetchall():
    print(row)
conn.close()
import pandas as pd
import chromadb
import os
import shutil
from sentence_transformers import SentenceTransformer

# ── Enrichment ─────────────────────────────────────────────────────────────────
def enrich_data(df):
    df.columns = df.columns.str.strip()
    df["Revenue"] = df["Units Sold"] * df["Price"] * (1 - (df["Discount"] / 100))
    df["is_stockout"] = df["Inventory Level"] < df["Demand"]

    def build_document(row):
        date_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
        lost = max(int(row["Demand"]) - int(row["Inventory Level"]), 0)
        doc = (
            f"Date: {date_str} | "
            f"Store: {row['Store ID']} | "
            f"Product: {row['Product ID']} | "
            f"Category: {row['Category']} | "
            f"Region: {row['Region']} | "
            f"Units Sold: {int(row['Units Sold'])} | "
            f"Demand: {int(row['Demand'])} | "
            f"Inventory Level: {int(row['Inventory Level'])} | "
            f"Revenue: ${round(row['Revenue'], 2)} | "
            f"Price: ${row['Price']} | "
            f"Discount: {row['Discount']}% | "
            f"Promotion: {'Yes' if row['Promotion'] == 1 else 'No'} | "
            f"Weather: {row['Weather Condition']} | "
            f"Seasonality: {row['Seasonality']} | "
        )
        if row["is_stockout"]:
            doc += f"Status: STOCKOUT — Lost demand of {lost} units."
        else:
            doc += "Status: Healthy stock."
        return doc

    df["Document"] = df.apply(build_document, axis=1)
    return df


# ── Setup ──────────────────────────────────────────────────────────────────────
DB_PATH = "./retail_vector_db"
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print("Old vector DB removed. Rebuilding...")

print("Reading and enriching data...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data", "sales_data.csv")
df = pd.read_csv(csv_path)
df = enrich_data(df)

# ── ChromaDB ───────────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="inventory_logs",
    metadata={"hnsw:space": "cosine"}
)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Batched Insertion ──────────────────────────────────────────────────────────
batch_size = 1000
total_rows = len(df)
print(f"Indexing {total_rows} enriched records...")

for i in range(0, total_rows, batch_size):
    batch_df = df.iloc[i: i + batch_size]
    ids        = [f"id_{idx}" for idx in batch_df.index]
    documents  = batch_df["Document"].tolist()
    embeddings = embed_model.encode(documents, show_progress_bar=False).tolist()
    metadatas = [
        {
            "store_id":    str(row["Store ID"]),
            "product_id":  str(row["Product ID"]),
            "category":    str(row["Category"]),
            "region":      str(row["Region"]),
            "date":        str(pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")),
            "revenue":     float(row["Revenue"]),
            "is_stockout": bool(row["is_stockout"]),
            "promotion":   bool(row["Promotion"] == 1),
        }
        for _, row in batch_df.iterrows()
    ]
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed batch {i // batch_size + 1}: rows {i}–{min(i + batch_size, total_rows)}")

print(f"\nDone. Total records indexed: {collection.count()}")
print("\n--- SAMPLE DOCUMENT ---")
sample = collection.get(ids=["id_0"])
print(sample["documents"][0])
print("\n--- SAMPLE METADATA ---")
print(sample["metadatas"][0])
import pandas as pd
import chromadb
import os
import shutil
from sentence_transformers import SentenceTransformer

# 1. ENRICHMENT LOGIC
def enrich_data(df):
    # 1. Standardize column names (case-insensitive check)
    df.columns = df.columns.str.strip() # Removes accidental spaces
    
    # 2. Calculate metrics (Using dashboard-consistent logic)
    df["Revenue"] = df["Units Sold"] * df["Price"] * (1 - (df["Discount"] / 100))
    df["is_stockout"] = df["Inventory Level"] < df["Demand"]
    
    # 3. Create the Enriched 'Document'
    df['Document'] = (
        f"Date: {pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')} | "
        f"Store: {df['Store ID']} | "
        f"Category: {df['Category']} | "
        f"Revenue: ${df['Revenue'].round(0)} | "
        + df.apply(lambda x: f"STOCKOUT ALERT: Lost demand of {x['Demand'] - x['Inventory Level']} units." 
                   if x['is_stockout'] else "Status: Healthy.", axis=1)
    )
    return df
# 2. SETUP
DB_PATH = "./retail_vector_db"
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH) # Fresh start to ensure logic consistency

print("Reading and enriching data...")
# The ../ tells Python to go "up one level" from the rag folder
df = pd.read_csv("../data/sales_data.csv")
df = enrich_data(df)

# 3. INDEXING SETUP
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="inventory_logs", 
    metadata={"hnsw:space": "cosine"}
)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. BATCHED INSERTION
batch_size = 1000
total_rows = len(df)
print(f"Indexing {total_rows} enriched records...")

for i in range(0, total_rows, batch_size):
    batch_df = df.iloc[i : i + batch_size]
    
    # Prepare data for ChromaDB
    ids = [f"id_{idx}" for idx in batch_df.index]
    documents = batch_df['Document'].tolist()
    embeddings = model.encode(documents, show_progress_bar=False).tolist()
    
    # Enriched Metadata
    metadatas = [
        {
            "store_id": str(row['Store ID']),
            "category": str(row['Category']),
            "revenue": float(row['Revenue']),
            "is_stockout": bool(row['is_stockout'])
        } 
        for _, row in batch_df.iterrows()
    ]
        
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"Indexed batch {i // batch_size + 1}: Rows {i} to {min(i + batch_size, total_rows)}")

print(f"\nDatabase complete. Total records: {collection.count()}")
"""
Builds a fresh ChromaDB vector database from the RetailCo policy PDF only.
Any previous retail-data collection (inventory_logs) is removed, since RAG
no longer answers retail-data queries -- only unstructured policy/SOP
knowledge that isn't stored as rows in Master_View.
"""

import os
import re
import shutil

import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
DB_PATH        = "./retail_vector_db"
PDF_PATH       = "./RetailCo_Policy_Handbook.pdf"
COLLECTION     = "company_policies"
EMBED_MODEL    = "all-MiniLM-L6-v2"

# Keyword -> section_number map. Used later at query time to apply metadata
# filtering (e.g. a question containing "refund" is routed to section 6).
# Built once here and saved alongside the DB so the retrieval script can
# derive it directly from section titles too, but kept explicit for clarity.
SECTION_KEYWORDS = {
    1:  ["company overview", "overview", "how many stores", "who is the ceo", "ceo"],
    2:  ["organizational structure", "org structure", "manager", "reporting", "hierarchy"],
    3:  ["store operating", "store hours", "opening hours", "safety stock", "stockout"],
    4:  ["warehouse", "inventory policy", "replenishment", "reorder", "seasonal stock"],
    5:  ["pricing", "promotions", "competitor pricing", "discount policy"],
    6:  ["return", "refund", "exchange", "damaged goods", "clearance", "sale item"],
    7:  ["customer service", "complaint", "feedback"],
    8:  ["leave", "attendance", "sick leave", "paid leave", "absence"],
    9:  ["expense", "reimbursement", "claim"],
    10: ["vendor", "procurement", "purchase order", "supplier", "onboarding"],
    11: ["it policy", "data security", "data breach", "access control", "role based"],
    12: ["health and safety", "fire safety", "protective equipment", "workplace incident"],
    13: ["escalation", "escalation matrix", "who do i contact", "who handles"],
}


def extract_sections(pdf_path: str):
    """Extract text from the PDF and split it into one chunk per numbered section."""
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Split right before each "N. Heading" pattern
    raw_chunks = re.split(r"(?=^\d+\.\s)", full_text, flags=re.MULTILINE)

    sections = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        match = re.match(r"^(\d+)\.\s+(.+?)\n", chunk)
        if not match:
            continue  # skip the title/header block before section 1
        section_number = int(match.group(1))
        section_title = match.group(2).strip()
        sections.append(
            {
                "section_number": section_number,
                "section_title": section_title,
                "text": chunk,
            }
        )
    return sections


def build_vector_db():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("Old vector DB removed (including any retail-data collections).")

    print("Extracting sections from PDF...")
    sections = extract_sections(PDF_PATH)
    print(f"Found {len(sections)} sections.")

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    embed_model = SentenceTransformer(EMBED_MODEL)

    ids        = [f"section_{s['section_number']}" for s in sections]
    documents  = [s["text"] for s in sections]
    metadatas  = [
        {
            "section_number": s["section_number"],
            "section_title": s["section_title"],
            "keywords": ", ".join(SECTION_KEYWORDS.get(s["section_number"], [])),
        }
        for s in sections
    ]

    print("Embedding and indexing sections...")
    embeddings = embed_model.encode(documents, show_progress_bar=False).tolist()
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    print(f"\nDone. Total sections indexed: {collection.count()}")
    print("\n--- SAMPLE DOCUMENT ---")
    sample = collection.get(ids=["section_6"])
    print(sample["documents"][0])
    print("\n--- SAMPLE METADATA ---")
    print(sample["metadatas"][0])


if __name__ == "__main__":
    build_vector_db()
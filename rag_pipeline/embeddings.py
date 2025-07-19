import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
DATA_PATH = "data/cleaned_data.json"
INDEX_PATH = "rag_pipeline/vectors/loan_vectors.faiss"
CHUNKS_PATH = "rag_pipeline/vectors/chunk_texts.npy"
MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight but effective model

def main():
    # Create directories if needed
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    
    # Load embedding model
    print(f"🔧 Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Load data
    print(f"📖 Loading data from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Process embeddings
    embeddings = []
    chunk_texts = []
    
    print(f"🔍 Generating embeddings for {len(data)} items...")
    for item in tqdm(data, desc="Processing"):
        text = item.get("content", "")
        if text and len(text.strip()) > 0:
            # Generate embedding
            emb = model.encode(text, convert_to_numpy=True).astype("float32")
            embeddings.append(emb)
            chunk_texts.append(text)
    
    if not embeddings:
        raise ValueError("No valid embeddings were generated - check your input data")
    
    # Create and save FAISS index
    print("💾 Saving vector database...")
    embeddings_np = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    faiss.write_index(index, INDEX_PATH)
    
    # Save chunk texts
    np.save(CHUNKS_PATH, np.array(chunk_texts, dtype=object))
    
    print(f"""
✅ Successfully created:
- FAISS index: {INDEX_PATH}
- Text chunks: {CHUNKS_PATH}
- Total embeddings: {len(embeddings)}
""")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Possible solutions:")
        print("1. Check data/cleaned_data.json exists and contains valid content")
        print("2. Verify sentence-transformers is installed (pip install sentence-transformers)")
        print("3. Ensure you have enough disk space")
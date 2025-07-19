import os
import numpy as np
import faiss
import time
from typing import List
from sentence_transformers import SentenceTransformer

# Configuration
TOP_K = 3
VECTORS_DIR = "rag_pipeline/vectors"
CHUNKS_PATH = os.path.join(VECTORS_DIR, "chunk_texts.npy")
INDEX_PATH = os.path.join(VECTORS_DIR, "loan_vectors.faiss")

class QueryProcessor:
    def __init__(self):
        """Initialize with FAISS index and chunk data"""
        os.makedirs(VECTORS_DIR, exist_ok=True)

        try:
            self.chunks = np.load(CHUNKS_PATH, allow_pickle=True)
            self.index = faiss.read_index(INDEX_PATH)
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Successfully loaded FAISS index, chunk texts, and embedding model")
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            raise

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Convert input query into embedding vector"""
        return self.model.encode(query, convert_to_numpy=True).astype("float32")

    def retrieve_relevant_chunks(self, query: str, top_k: int = TOP_K) -> List[str]:
        """Retrieve top-k most similar chunks using FAISS"""
        query_vector = self.get_query_embedding(query)
        try:
            D, I = self.index.search(np.array([query_vector]), top_k)
            return [str(self.chunks[i]) for i in I[0] if i < len(self.chunks)]
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return []

    def generate_answer(self, query: str) -> str:
        """Simulate answer using retrieved context (no LLM)"""
        context_chunks = self.retrieve_relevant_chunks(query)
        if not context_chunks:
            return "I couldn't find relevant information to answer your question."

        context = "\n\n".join(
            f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
        )

        return f"""Based on the following context, the answer to your question is:

{context}

(Note: This is a simulated local answer. Integrate a local LLM if needed.)"""

def main():
    try:
        processor = QueryProcessor()
        print("\n💬 Loan Product Assistant is ready. Type 'exit' to quit.\n")

        while True:
            query = input("\n❓ Ask your question: ").strip()
            if query.lower() in ('exit', 'quit'):
                print("👋 Exiting. Goodbye!")
                break

            if not query:
                print("⚠️ Please enter a valid question.")
                continue

            start = time.time()
            answer = processor.generate_answer(query)
            print(f"\n💡 Answer (generated in {time.time() - start:.2f} seconds):\n{answer}")

    except Exception as e:
        print(f"❌ Failed to initialize: {str(e)}")

if __name__ == "__main__":
    main()

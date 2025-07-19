import os
import time
import faiss
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer

# Load OpenRouter key
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Config
INDEX_PATH = "rag_pipeline/vectors/loan_vectors.faiss"
CSV_PATH = "data/cleaned_data.csv"
MODEL_CHAT = "openai/gpt-3.5-turbo"
TEMPERATURE = 0.2
TOP_K = 3
TIMEOUT = 30

class LoanQueryProcessor:
    def __init__(self):
        print("✅ Initializing LoanQueryProcessor...")
        try:
            self.index = faiss.read_index(INDEX_PATH)
            self.df = pd.read_csv(CSV_PATH)
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Loaded FAISS index, cleaned CSV, and local embedding model")
        except Exception as e:
            print(f"❌ Initialization error: {str(e)}")
            raise

    def get_query_embedding(self, query: str) -> np.ndarray:
        return self.embedder.encode(query, convert_to_numpy=True).astype("float32")

    def retrieve_chunks(self, query_embedding: np.ndarray) -> List[str]:
        try:
            D, I = self.index.search(query_embedding.reshape(1, -1), TOP_K)
            return [self.df.iloc[i]["content"] for i in I[0] if i < len(self.df)]
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []

    def call_openrouter(self, query: str, context: str) -> str:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Loan Product Assistant",
            "Content-Type": "application/json"
        }

        messages = [
            {"role": "system", "content": "You are a helpful assistant for Bank of Maharashtra's loan products. Answer only from context."},
            {"role": "user", "content": f"""Context:\n{context}\n\nQuestion: {query}\n\nGive a short and precise answer. Mention which context(s) were used (e.g., [Context 1]). Say 'I don’t know' if not found in context."""}
        ]

        payload = {
            "model": MODEL_CHAT,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": 500
        }

        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=TIMEOUT)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ OpenRouter call failed: {e}")
            return "Sorry, I couldn't generate an answer."

    def answer_query(self, query: str) -> str:
        start = time.time()

        embedding = self.get_query_embedding(query)
        chunks = self.retrieve_chunks(embedding)
        if not chunks:
            return "No relevant information found."

        context = "\n\n".join(f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(chunks))
        answer = self.call_openrouter(query, context)

        print(f"⏱️ Generated in {time.time() - start:.2f}s")
        return answer

def main():
    print("💬 Bank of Maharashtra Loan Assistant is ready. Type 'exit' to quit.")
    try:
        processor = LoanQueryProcessor()
        while True:
            query = input("\n❓ Your question: ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                print("⚠️ Please enter a question.")
                continue
            print("\n💡 Answer:\n" + processor.answer_query(query))
    except Exception as e:
        print(f"❌ Startup failed: {e}")

if __name__ == "__main__":
    main()

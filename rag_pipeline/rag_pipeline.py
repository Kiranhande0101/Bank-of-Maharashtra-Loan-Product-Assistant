import requests
import numpy as np
import faiss
import pandas as pd
import time
from typing import Optional, List

# Configuration
API_KEY = "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"  # Replace with your actual key
EMBEDDING_API_URL = "https://openrouter.ai/api/v1/embeddings"
TEXT_GEN_API_URL = "https://openrouter.ai/api/v1/chat/completions"
FAISS_INDEX_PATH = "rag_pipeline/vectors/loan_vectors.faiss"
DATA_PATH = "data/cleaned_data.csv"
MODEL_EMBEDDING = "openai/text-embedding-3-small"
MODEL_CHAT = "openai/gpt-3.5-turbo"  # or "openai/gpt-4"
TEMPERATURE = 0.2
TOP_K = 3
TIMEOUT = 30

class LoanQueryProcessor:
    def __init__(self):
        """Initialize the query processor with loaded index and data"""
        try:
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            self.df = pd.read_csv(DATA_PATH)
            print("✅ Successfully loaded FAISS index and data")
        except Exception as e:
            print(f"❌ Failed to load data: {str(e)}")
            raise

    def _make_api_request(self, url: str, payload: dict, max_retries: int = 3) -> Optional[dict]:
        """Generic API request with retry logic"""
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Loan Product Assistant",
            "Content-Type": "application/json"
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=TIMEOUT
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        print(f"❌ Failed after {max_retries} attempts")
        return None

    def get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding vector for a query"""
        payload = {
            "model": MODEL_EMBEDDING,
            "input": query
        }

        result = self._make_api_request(EMBEDDING_API_URL, payload)
        if result and "data" in result:
            return np.array(result["data"][0]["embedding"]).astype("float32")
        return None

    def retrieve_relevant_text(self, query_embedding: np.ndarray) -> List[str]:
        """Retrieve most relevant text chunks"""
        try:
            D, I = self.index.search(query_embedding.reshape(1, -1), TOP_K)
            return [self.df.iloc[int(i)]['content'] for i in I[0] if i < len(self.df)]
        except Exception as e:
            print(f"❌ Search failed: {str(e)}")
            return []

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate an answer using RAG pipeline"""
        context = "\n\n".join(
            f"[Context {i+1}]: {chunk}"
            for i, chunk in enumerate(context_chunks)
        )

        messages = [{
            "role": "system",
            "content": "You are a helpful loan assistant for Bank of Maharashtra. Answer questions based on the provided context."
        }, {
            "role": "user",
            "content": f"""Context:
{context}

Question: {query}

Provide a concise answer mentioning which context(s) you used (e.g. [Context 1]). 
If the answer isn't in the context, say you don't know."""
        }]

        payload = {
            "model": MODEL_CHAT,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": 500
        }

        result = self._make_api_request(TEXT_GEN_API_URL, payload)
        if result and "choices" in result:
            return result["choices"][0]["message"]["content"]
        return "Sorry, I couldn't generate an answer at the moment."

    def answer_query(self, query: str) -> str:
        """Full pipeline to answer a query"""
        start_time = time.time()
        
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return "I couldn't process your question at the moment."
        
        # Retrieve relevant content
        context_chunks = self.retrieve_relevant_text(query_embedding)
        if not context_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Generate answer
        answer = self.generate_answer(query, context_chunks)
        
        elapsed = time.time() - start_time
        print(f"\nGenerated in {elapsed:.2f} seconds")
        return answer

def main():
    try:
        processor = LoanQueryProcessor()
        print("\nBank of Maharashtra Loan Assistant ready. Type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("\n❓ Ask your question: ").strip()
                if query.lower() in ('exit', 'quit'):
                    break
                
                if not query:
                    print("⚠️ Please enter a valid question")
                    continue
                
                answer = processor.answer_query(query)
                print(f"\n💡 Answer:\n{answer}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
                
    except Exception as e:
        print(f"❌ Failed to initialize query processor: {str(e)}")

if __name__ == "__main__":
    main()
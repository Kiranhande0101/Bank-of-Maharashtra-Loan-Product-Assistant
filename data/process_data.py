import os
import numpy as np
import faiss
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = "https://openrouter.ai/api/v1/embeddings"
data_path = "data/loan_data.txt"  # ✅ Corrected path

def main():
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    vectors = []
    for chunk in chunks:
        response = requests.post(
            base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://yourapp.com",
                "X-Title": "Loan Product Assistant"
            },
            json={
                "model": "openai/text-embedding-3-small",
                "input": chunk
            }
        )
        result = response.json()
        vectors.append(result["data"][0]["embedding"])

    vectors = np.array(vectors).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    os.makedirs("rag_pipeline/vectors", exist_ok=True)
    faiss.write_index(index, "rag_pipeline/vectors/loan_vectors.faiss")
    np.save("rag_pipeline/vectors/chunk_texts.npy", np.array(chunks))

    print("✅ Embeddings generated and saved (via OpenRouter)")

if __name__ == "__main__":
    main()

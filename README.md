#  Bank of Maharashtra – Loan Product Assistant (RAG Pipeline)

This project is a **Retrieval-Augmented Generation (RAG) based Loan Assistant** that helps answer questions related to various loan products of **Bank of Maharashtra**. It uses **FAISS** for similarity search and **OpenRouter** (via GPT models) for generating answers based on real document context.

---

##  Project Structure

```
loan-assistant/
├── data/
│   └── cleaned_data.csv                # Pre-processed loan document text
├── rag_pipeline/
│   ├── query_processor_openrouter.py  # Main RAG pipeline using OpenRouter
│   └── vectors/
│       └── loan_vectors.faiss         # FAISS vector store
├── .env                                   # Secret API keys (not committed)
├── requirements.txt
└── README.md                          # This file
```

---

##  Features

*  Local document-based Q\&A
*  OpenRouter API with GPT-3.5/4
*  Vector search using FAISS
*  Clean Pandas DataFrame-based context lookup
*  OpenAI-compatible embeddings via OpenRouter

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/kiranhande0101/loan-assistant.git](https://github.com/Kiranhande0101/Bank-of-Maharashtra-Loan-Product-Assistant
cd loan-assistant
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your `.env` File

Create a `.env` file in the root with the following:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```


##  Data Format

The `data/cleaned_data.csv` file must contain the following column:

* `content` → Text chunk from scraped loan documents.

---

##  Run the Assistant

```bash
python rag_pipeline/query_processor_openrouter.py
```

You’ll see:

```bash
 Bank of Maharashtra Loan Assistant is ready. Type 'exit' to quit.
❓ Your question:
```

Type your question like:

```
What is the interest rate for car loan?
```

And it will return an answer like:

```
💡 Answer:
Car loan interest rate at Bank of Maharashtra starts from 7.70% to 12.00%, based on RLLR (Repo Linked Lending Rate) and credit score of the customer. [Context 1]
```

---

##  Example Questions

* What is the interest rate for car loan?
* How much loan can I get for home loan?
* What documents are required for personal loan?
* What is the maximum tenure for education loan?

---

## Technologies Used

* `Python 3.10+`
* `OpenRouter API`
* `Sentence Transformers` (for embeddings)
* `FAISS` (for vector similarity search)
* `Pandas`
* `dotenv` (for environment config)


## Author

**Kiran Hande**

 
 

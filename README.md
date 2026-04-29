# StudentCompass

StudentCompass is an academic information assistant designed for Metropolitan State University.  
It provides fast, context‑aware answers sourced from official university documents using a Retrieval‑Augmented Generation (RAG) backend and a React frontend.

---

## 📚 Documentation

Full documentation lives at the root of this repository:

- **Developer Guide** — backend architecture, ingestion pipeline, query engine, evaluation system  
  `developer_guide.md`

- **End User Guide** — how to use the chat, Admin page, and Test page  
  `end_user_guide.md`

These guides contain all detailed explanations, diagrams, and instructions.

---

## 🚀 Quick Start

### Backend (Flask + RAG Engine)

```bash
cd backend/rag
pip install -r requirements.txt
python gcs_upload.py

Backend runs at:  
`http://localhost:5000`

---

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev

Frontend runs at:  
`http://localhost:5173`  
Vite automatically proxies API calls to the backend during development.

---

## 🧠 Features

- RAG-based question answering using ChromaDB + Gemini 2.5 Flash  
- Document ingestion pipeline (chunking, embedding, metadata storage)  
- Streaming responses (SSE) for real-time token output  
- Admin interface for uploading, replacing, deleting, and syncing documents  
- Built‑in evaluation tools (browser-based + Optuna/RAGAS offline script)

---

## 🏗️ Tech Stack

**Backend**
- Python (Flask)
- LlamaIndex (core, embeddings, vector stores)
- ChromaDB
- Google Generative AI (Gemini 2.5 Flash)
- LangChain (evaluation helpers)
- Optuna + RAGAS (offline hyperparameter search)

**Frontend**
- React 18
- Vite
- Tailwind CSS

**Infrastructure**
- Google Cloud Storage (document storage)

---

## 📁 Repository Structure
backend/
rag/
.env
eval_optuna.py
gcs_upload.py
ingest.py
query.py
gold_questions.json
optuna_results.json
service_account.json
requirements.txt
frontend/
developer_guide.md
end_user_guide.md


---

## 👥 Contributors

- Cheng  
- Javier
- Hana

---

## 📄 License

This project is licensed under the MIT License.


# FastAPI: keeps everything loaded in memory
# command to run main.py: uvicorn main:app --reload

import os

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from chromadb import PersistentClient

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()
app = FastAPI()

# load everything once at startup
# 1. embedding model
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. LLM
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

# 3. chroma vector store
chroma_client = PersistentClient(path="rag/chroma")
chroma_collection = chroma_client.get_or_create_collection("studentcompass")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 4. storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 5. vector index
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)

# 6. query engine
query_engine = index.as_query_engine()


# request model
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

# API endpoint
@app.post("/query")
def query(req: QueryRequest):
    engine = index.as_query_engine(similarity_top_k=req.top_k)
    response = engine.query(req.question)

    sources = []
    for node in response.source_nodes:
        metadata = node.node.metadata

        source = (
            metadata.get("file_path")
            or metadata.get("url")
            or metadata.get("source")
            or "Unknown source"
        )

        doc_type = metadata.get("doc_type", "Unknown type")
        summary = metadata.get("summary", None)

        sources.append({
            "source": source,
            "doc_type": doc_type,
            "summary": summary[:150] + "..." if summary else None
        })

    return {
        "answer": str(response),
        "sources": sources
    }






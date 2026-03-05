# load the vector store and answer questions

import sys
import os

from chromadb import PersistentClient
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.prompts import PromptTemplate

load_dotenv()


# configure Vertex AI LLM using environment variables
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",   # fast and cost-effective Gemini model
    api_key=os.getenv("GEMINI_API_KEY")
)

# prompt template
qa_prompt = PromptTemplate(
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer:"
)

# query function
def run_query(question: str):
    # 1. load chromadb persistent storage
    chroma_client = PersistentClient(path="chroma")
    chroma_collection = chroma_client.get_or_create_collection("studentcompass")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2. load embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. load the index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # 4. create a query engine
    query_engine = index.as_query_engine(
        response_mode="compact",
        text_qa_template=qa_prompt,
        similarity_top_k=3,
        use_async=False,
        streaming=False
    )

    # 5. ask the question
    response = query_engine.query(question)
    print("\nQuestion:")
    print(question)
    print("\nAnswer:")
    print(response)
    print("\nSources:")
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

        print(f"- Source: {source}")
        print(f"  Type: {doc_type}")

        if summary:
            print(f"  Summary: {summary[:150]}...")  #print first 150 characters
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # if no argument provided, ask for input interactively
        question = input("Enter your question: ")
    else:
        question = " ".join(sys.argv[1:])

    run_query(question)

# turn documents into embeddings and store them in chromaDB

import os

from chromadb import PersistentClient
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

load_dotenv()


# configure Vertex AI LLM using environment variables
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

def run_ingestion():
    # 1. load documents from data folder and from URLs
    documents = SimpleDirectoryReader(
        input_dir="data",
        recursive=True,
        required_exts=[".txt", ".pdf", ".docx"]
    ).load_data()

    urls = [
        "https://www.factretriever.com/bird-facts",
        "https://www.metrostate.edu/about/mission",
        # add more URLS here
    ]

    if urls:
        url_docs = TrafilaturaWebReader().load_data(urls)
        documents.extend(url_docs)

    # display each loaded document and its assigned metadata.
    # this helps verify that all file types (.txt, .docx, .pdf, URLs) were detected correctly
    # and that folder-based doc_type classification is working as expected.
    print("\nLoaded documents:")
    for doc in documents:
        path = doc.metadata.get("file_path")

        # URL documents
        if path is None:
            path = doc.metadata.get("url") or doc.metadata.get("source") or "Unknown URL"
            doc.metadata["source_type"] = "url"
            doc.metadata["doc_type"] = "web_page"

        else:
            # local documents
            doc.metadata["source_type"] = "local_file"

            # extract folder name as doc_type
            folder = os.path.basename(os.path.dirname(path))
            doc.metadata["doc_type"] = folder

        print(f"- {path}  (doc_type: {doc.metadata['doc_type']})")

    CONCEPTUAL_TYPES = {
        "admissions",
        "financial_aid",
        "graduation",
        "policies",
        "registration",
        "student_support",
        "tuition_fees"
    }

    print("\nGenerating document summaries (conceptual docs only)...")

    for doc in documents:
        doc_type = doc.metadata.get("doc_type")

        if doc_type in CONCEPTUAL_TYPES:
            prompt = f"Summarize this document in 4–6 sentences. Focus on the main themes, categories, and purpose.\n\nDocument:\n{doc.text}"

            summary = Settings.llm.complete(prompt).text.strip()
            doc.metadata["summary"] = summary

            print(f"✓ Summary added for {doc_type} document")
        else:
            print(f"- Skipped summary for {doc_type} (not conceptual)")

    # 2. create embeddings (convert text into vectors)
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. set up ChromaDB persistent storage
    chroma_client = PersistentClient(path="chroma")
    chroma_collection = chroma_client.get_or_create_collection("studentcompass")
    print("\nBefore ingestion, collection has:", chroma_collection.count(), "embeddings")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

    # 4. take documents and split them into chunks
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[parser],
        metadata_mode=MetadataMode.ALL
    )

    print("After ingestion, collection has:", chroma_collection.count(), "embeddings")
    print("Ingestion complete! Vector store saved to /chroma")

if __name__ == "__main__":
    run_ingestion()

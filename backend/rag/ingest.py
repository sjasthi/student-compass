# turn documents into embeddings and store them in chromaDB

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb import PersistentClient

def run_ingestion():
    # 1. load documents from data folder
    documents = SimpleDirectoryReader("data").load_data()

    # print out the type of documents it took
    # testing to see if it will take .txt, .docx, and .pdf files
    print("\nLoaded documents:")
    for doc in documents:
        path = doc.metadata.get("file_path", "unknown")
        ext = path.split(".")[-1].lower()
        print(f"- {path}  (type: {ext})")

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

    # 4. take documents and split them into chunks
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    print("After ingestion, collection has:", chroma_collection.count(), "embeddings")

    print("Ingestion complete! Vector store saved to /chroma")

if __name__ == "__main__":
    run_ingestion()
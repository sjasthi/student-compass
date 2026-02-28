# TODO: Future ingestion improvements
#1. metadata extraction: help retriever understand document type (more accurate search)
#2. document summaries: give high-level meaning to each doc (better answers for broad questions)
#3. chunk metadata: add structure to each chunk (more trustworthy and grounded answer, reference source of info)
#4. chunk preview: catch junk text early (cleaner ingestion, fewer bugs)

# configure Vertex AI LLM using environment variables

Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=""
)

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

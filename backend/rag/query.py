# load the vector store and answer questions

import sys
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb import PersistentClient
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate


# configure the LLM
Settings.llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    context_window=2048,
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
        #disable refinement to reduce noise
        use_async=False,
        streaming=False
    )

    # 5. ask the question
    response = query_engine.query(question)

    # check for accuracy, if there is no data then print no information
    retrieved_nodes = query_engine.retrieve(question)
    if len(retrieved_nodes) == 0:
        return "I don’t have information about that in my documents."
    print("\nQuestion:")
    print(question)
    print("\nAnswer:")
    print(response)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    run_query(question)
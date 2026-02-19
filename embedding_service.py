from sentence_transformers import SentenceTransformer
import chromadb

class EmbeddingService:
    def __init__(self):
        print("Loading embedding model...")
        # Load the local embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Initializing ChromaDB...")
        # Initialize ChromaDB - simpler, modern approach
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="student_compass_docs"
        )
        print("Ready!")
    
    def create_embeddings(self, texts):
        """Convert texts to embeddings"""
        return self.model.encode(texts).tolist()
    
    def store_embeddings(self, chunks, metadata=None):
        """Store chunks and their embeddings in ChromaDB"""
        embeddings = self.create_embeddings(chunks)
        
        # Generate unique IDs for each chunk
        ids = [f"doc_{i}_{hash(chunk)}" for i, chunk in enumerate(chunks)]
        
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids,
            metadatas=metadata or [{"source": "unknown"}] * len(chunks)
        )
    
    def query_embeddings(self, query_text, n_results=5):
        """Search for similar documents"""
        query_embedding = self.create_embeddings([query_text])
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results

from fastapi import FastAPI, UploadFile, File, HTTPException
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
import tempfile
import os

app = FastAPI(title="StudentCompass Document Embeddings API")

# Initialize services
print("Initializing services...")
doc_processor = DocumentProcessor()
embedding_service = EmbeddingService()

@app.get("/")
async def root():
    return {"message": "StudentCompass Embeddings API is running!"}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (PDF or DOCX)"""
    
    # Validate file type
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load document based on type
        if file.filename.endswith('.pdf'):
            text = doc_processor.load_pdf(tmp_path)
        else:
            text = doc_processor.load_docx(tmp_path)
        
        # Chunk the text
        chunks = doc_processor.chunk_text(text)
        
        # Create and store embeddings
        metadata = [{"filename": file.filename, "chunk_id": i} for i in range(len(chunks))]
        embedding_service.store_embeddings(chunks, metadata)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "total_characters": len(text)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

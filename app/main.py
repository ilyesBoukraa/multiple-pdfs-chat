from pinecone import Pinecone
from pydantic import BaseModel
from typing import List, Union
from fastapi import FastAPI, File, HTTPException, UploadFile
from app.core.config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME
from app.pre_processing.pdf_to_text import my_pdfs_to_text
from app.pre_processing.text_chunking import my_text_chunking
from app.pre_processing.vectorization import my_vectorization

app = FastAPI()

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/upload-pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")
    
    raw_text = my_pdfs_to_text(files)
    
    text_chunks = my_text_chunking(raw_text)
     
    my_vectorization(text_chunks, index)

    return {"message": "PDFs processed successfully", "num_chunks": len(text_chunks)}
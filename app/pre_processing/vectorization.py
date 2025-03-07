from typing import List
import uuid
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from pinecone import Pinecone

def my_vectorization(text_chunks: List[str], index: Pinecone.Index):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    vectors = []
    for text in text_chunks:
        vector = embeddings.embed_query(text)  
        vectors.append((str(uuid.uuid4()), vector, {"text": text}))  # Unique ID, vector, metadata

    # Upload to Pinecone
    index.upsert(vectors)
    return {"message": "Embeddings stored in Pinecone", "num_vectors": len(vectors)}
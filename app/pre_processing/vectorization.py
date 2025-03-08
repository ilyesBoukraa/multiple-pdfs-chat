from typing import List
import uuid
# from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
#  from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings



# from langchain.vectorstores import FAISS
from pinecone import Pinecone

def my_vectorization(text_chunks: List[str], index: Pinecone.Index):
    hf = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", 
                                       model_kwargs={"device": "cpu"}) 
    vectors = []
    for text in text_chunks:
        vector = hf.embed_query(text)  
        vectors.append((str(uuid.uuid4()), vector, {"text": text}))  # Unique ID, vector, metadata

    # Upload to Pinecone
    index.upsert(vectors)
    return {"message": "Embeddings stored in Pinecone", "num_vectors": len(vectors)}
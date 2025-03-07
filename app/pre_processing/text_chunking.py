from typing import List
from langchain.text_splitter import CharacterTextSplitter

def my_text_chunking(text: List[str])-> List[str]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


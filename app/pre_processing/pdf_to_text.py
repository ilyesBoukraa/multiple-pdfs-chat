from typing import List
from PyPDF2 import PdfReader
from fastapi import UploadFile

def my_pdfs_to_text(pdfs: List[UploadFile])-> str:
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf.file)
        for page in reader.pages:
            text += page.extract_text()
    return text
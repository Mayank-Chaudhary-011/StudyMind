# Document Loader
import os
from langchain_community.document_loaders import PyPDFLoader



def load_docs(folder_path="data", file_paths=None):
    num_docs = 0
    all_docs = []

    if file_paths is None:
        file_paths = [
            os.path.join(folder_path, filename)
            for filename in os.listdir(folder_path)
            if filename.lower().endswith(".pdf")
        ]

    for pdf_path in file_paths:
        if not pdf_path.lower().endswith(".pdf"):
            continue

        loader = PyPDFLoader(pdf_path)
        doc = loader.load()
        all_docs.extend(doc)
        num_docs += 1

    print("Total pdfs :" , num_docs)
    print("Total pages :" , len(all_docs))

    return all_docs

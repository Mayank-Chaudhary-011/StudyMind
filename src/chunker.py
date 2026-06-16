from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_docs(documents, chunk_size=512, chunk_overlap=50):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs

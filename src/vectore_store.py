import os
import uuid
import chromadb
from typing import List


class VectorStoreManager:
    def __init__(
        self,
        persist_directory="data/vector_store",
        collection_name="pdf_documents",
        persistent=True,
    ):
        """
        persistent=True  -> stores embeddings on disk at persist_directory
                             (shared across restarts / sessions)
        persistent=False -> in-memory Chroma client, isolated to this
                             Python object. Use this per Streamlit session
                             so different users never see each other's docs.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.persistent = persistent
        self.collection = None
        self.client = None
        self._initialize_store()

    def _initialize_store(self):
        if self.persistent:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Vector Store collection for PDF Embeddings in RAG",
                "hnsw:space": "cosine",
            }
        )
        print("Collection initialized:", self.collection_name)

    def add_documents(self, documents: List, embeddings, replace: bool = False):
        """
        Add chunks + embeddings to the collection.

        replace=False (default) -> append to whatever is already indexed.
        replace=True            -> wipe the collection first, then add.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents doesn't match number of embeddings")

        if replace:
            existing = self.collection.get()
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                print("Cleared old data from collection")

        ids = []
        all_metadata = []
        documents_content = []
        embedding_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4()}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            all_metadata.append(metadata)

            documents_content.append(doc.page_content)
            embedding_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            metadatas=all_metadata,
            documents=documents_content,
            embeddings=embedding_list
        )

        print("Total documents added to vector store:", len(documents_content))
        print("Total docs in collection:", self.collection.count())

    def delete_by_source(self, source_path: str):
        """Remove every chunk that came from a specific source file path."""
        self.collection.delete(where={"source": source_path})
        print(f"Removed chunks for source: {source_path}")
        print("Total docs in collection:", self.collection.count())

    def get_sources(self):
        """Return the set of unique source file paths currently indexed."""
        existing = self.collection.get(include=["metadatas"])
        sources = set()
        for m in existing["metadatas"]:
            if m and "source" in m:
                sources.add(m["source"])
        return sources

    def count(self):
        return self.collection.count()
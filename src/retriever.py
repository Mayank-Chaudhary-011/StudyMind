class RAGRetriever:
    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def _get_anchor_chunks(self):
        """
        For each uploaded PDF (each unique 'source'), fetch its earliest
        chunk (lowest page, then lowest doc_index). This is almost always
        the title + start of the abstract, which is where things like
        acronym definitions ("RAG = Retrieval-Augmented Generation") tend
        to live but rarely get pulled in by pure semantic search.
        """
        try:
            sources = self.vector_store.get_sources()
        except Exception:
            return []

        anchors = []
        for source in sources:
            try:
                result = self.vector_store.collection.get(where={"source": source})
            except Exception:
                continue

            ids = result.get("ids", [])
            metadatas = result.get("metadatas", [])
            documents = result.get("documents", [])
            if not ids:
                continue

            best_idx = min(
                range(len(ids)),
                key=lambda i: (
                    metadatas[i].get("page", 0),
                    metadatas[i].get("doc_index", 0),
                )
            )
            anchors.append({
                "id": ids[best_idx],
                "metadata": metadatas[best_idx],
                "document": documents[best_idx],
                "distance": None,
                "similarity_score": None,
                "rank": None,
                "anchor": True,
            })
        return anchors

    def retrieve(self, query, top_k=5, score_threshold=-1.0):
        """
        score_threshold defaults to -1.0 (the lowest possible cosine
        similarity) so that, for short or ambiguous queries, the top_k
        results aren't silently discarded just because their similarity
        score happens to be low. Let the LLM decide relevance from context
        instead of filtering it out before it ever sees it.

        On top of the top_k semantic matches, each document's first chunk
        (title/abstract) is always included as an "anchor" chunk, since
        that's where definitions and acronym expansions usually live.
        """
        query_embedding = self.embedding_manager.generate_embedding([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]

            for i, (doc_id, metadata, document, distance) in enumerate(zip(ids, metadatas, documents, distances)):
                # With "hnsw:space": "cosine", distance is in [0, 2],
                # so similarity = 1 - distance is in [-1, 1].
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "metadata": metadata,
                        "document": document,
                        "distance": distance,
                        "similarity_score": similarity_score,
                        "rank": i + 1,
                        "anchor": False,
                    })
            print(f"Retrieved {len(retrieved_docs)} Documents")
        else:
            print("No documents found")

        # Add anchor chunks (doc_index == 0) that aren't already included.
        existing_ids = {d["id"] for d in retrieved_docs}
        anchors_added = 0
        for anchor in self._get_anchor_chunks():
            if anchor["id"] not in existing_ids:
                retrieved_docs.append(anchor)
                existing_ids.add(anchor["id"])
                anchors_added += 1

        if anchors_added:
            print(f"Added {anchors_added} anchor chunk(s)")

        return retrieved_docs
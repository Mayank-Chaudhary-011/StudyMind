from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Callable, Optional


class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        print("Loading Model.........", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        print("Embedding Dimensions --->", self.model.get_sentence_embedding_dimension())

    def generate_embedding(
        self,
        texts: List[str],
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Generate embeddings for a list of texts.

        If progress_callback is provided, it is called after every batch
        with (num_processed, total) so the UI can render a progress bar.
        """
        total = len(texts)
        if total == 0:
            return np.array([])

        all_embeddings = []
        for start in range(0, total, batch_size):
            batch = texts[start:start + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)

            if progress_callback:
                progress_callback(min(start + batch_size, total), total)

        embeddings = np.vstack(all_embeddings)
        print("Embedding Shape --->", embeddings.shape)
        return embeddings
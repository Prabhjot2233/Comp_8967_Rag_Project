from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initializes the embedding model.
        Supported models: 'BAAI/bge-small-en-v1.5' or 'intfloat/e5-base-v2'
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts.
        Returns a numpy array of shape (num_texts, embedding_dim).
        """
        if not texts:
            return np.empty((0, 0))
        
        # For E5 models, prefix query/passage text as required by E5 guidelines
        processed_texts = texts
        if "e5" in self.model_name.lower():
            # If we detect it's an E5 model, we prefix passages with 'passage: '
            # (queries should be prefixed with 'query: ' when embedding queries)
            processed_texts = [f"passage: {t}" if not t.startswith("passage:") and not t.startswith("query:") else t for t in texts]

        embeddings = self.model.encode(processed_texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates embedding for a single query string.
        """
        text_to_embed = query
        if "e5" in self.model_name.lower():
            text_to_embed = f"query: {query}"
        
        # Returns a 1D vector of shape (embedding_dim,)
        return self.embed_texts([text_to_embed])[0]

import faiss
import numpy as np
import os
from typing import Tuple, List

class FAISSIndexer:
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initializes FAISS Index.
        index_type can be: 'flat', 'ivf', 'hnsw'
        """
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.index = None
        self.doc_ids = []  # Maps FAISS internal ID to actual passage ID

    def build_index(self, embeddings: np.ndarray, doc_ids: List[str]):
        """
        Builds the FAISS index with the provided embeddings.
        """
        assert len(embeddings) == len(doc_ids), "Embeddings and doc_ids must have the same length"
        self.doc_ids = list(doc_ids)

        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype('float32'))
        
        elif self.index_type == "ivf":
            # IVF requires training. Number of centroids (nlist) can be proportional to sqrt(N)
            nlist = int(np.sqrt(len(embeddings))) if len(embeddings) > 0 else 1
            quantizer = faiss.IndexFlatL2(self.dimension)
            # Use IndexIVFFlat
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
            self.index.train(embeddings.astype('float32'))
            self.index.add(embeddings.astype('float32'))
            
        elif self.index_type == "hnsw":
            # HNSWFlat
            # M = 32 is a standard parameter for graph connectivity
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_L2)
            self.index.add(embeddings.astype('float32'))
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Searches the index for query embeddings.
        Returns:
            distances: numpy array of shape (num_queries, k)
            retrieved_doc_ids: List of List of original passage IDs
        """
        if self.index is None:
            raise ValueError("Index has not been built yet.")
        
        distances, indices = self.index.search(query_embeddings.astype('float32'), k)
        
        retrieved_ids = []
        for query_idx in range(len(query_embeddings)):
            query_retrieved = []
            for id_idx in indices[query_idx]:
                if id_idx != -1 and id_idx < len(self.doc_ids):
                    query_retrieved.append(self.doc_ids[id_idx])
                else:
                    query_retrieved.append(None)
            retrieved_ids.append(query_retrieved)
            
        return distances, retrieved_ids

    def save(self, file_path: str):
        """Saves the index and document mappings to disk."""
        if self.index is None:
            raise ValueError("No index to save.")
        faiss.write_index(self.index, file_path)
        # Save mapping
        with open(file_path + ".map", "w") as f:
            f.write("\n".join(self.doc_ids))

    def load(self, file_path: str):
        """Loads index and document mappings from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Index file not found: {file_path}")
        self.index = faiss.read_index(file_path)
        with open(file_path + ".map", "r") as f:
            self.doc_ids = f.read().splitlines()

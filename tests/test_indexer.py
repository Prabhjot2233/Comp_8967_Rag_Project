import os
import tempfile
import numpy as np
import pytest
from src.indexer import FAISSIndexer

def test_flat_indexer():
    dimension = 4
    indexer = FAISSIndexer(dimension=dimension, index_type="flat")
    
    # 3 dummy documents
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype='float32')
    doc_ids = ["doc_A", "doc_B", "doc_C"]
    
    indexer.build_index(embeddings, doc_ids)
    
    # Query very close to doc_B
    query = np.array([[0.1, 0.9, 0.0, 0.0]], dtype='float32')
    distances, retrieved_ids = indexer.search(query, k=2)
    
    assert retrieved_ids[0][0] == "doc_B"
    assert retrieved_ids[0][1] == "doc_A"

def test_ivf_indexer():
    dimension = 4
    indexer = FAISSIndexer(dimension=dimension, index_type="ivf")
    # Need slightly more embeddings to train IVF, let's create 10 embeddings
    embeddings = np.random.rand(10, dimension).astype('float32')
    doc_ids = [f"doc_{i}" for i in range(10)]
    
    indexer.build_index(embeddings, doc_ids)
    distances, retrieved_ids = indexer.search(embeddings[2:3], k=1)
    
    assert retrieved_ids[0][0] == "doc_2"

def test_hnsw_indexer():
    dimension = 4
    indexer = FAISSIndexer(dimension=dimension, index_type="hnsw")
    embeddings = np.random.rand(5, dimension).astype('float32')
    doc_ids = [f"doc_{i}" for i in range(5)]
    
    indexer.build_index(embeddings, doc_ids)
    distances, retrieved_ids = indexer.search(embeddings[3:4], k=1)
    
    assert retrieved_ids[0][0] == "doc_3"

def test_save_load_indexer():
    with tempfile.TemporaryDirectory() as tmpdir:
        dimension = 4
        indexer = FAISSIndexer(dimension=dimension, index_type="flat")
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype='float32')
        doc_ids = ["doc_A", "doc_B"]
        
        indexer.build_index(embeddings, doc_ids)
        
        index_file = os.path.join(tmpdir, "test_index.faiss")
        indexer.save(index_file)
        
        # Load into new indexer
        new_indexer = FAISSIndexer(dimension=dimension)
        new_indexer.load(index_file)
        
        query = np.array([[0.0, 1.0, 0.0, 0.0]], dtype='float32')
        distances, retrieved_ids = new_indexer.search(query, k=1)
        
        assert retrieved_ids[0][0] == "doc_B"

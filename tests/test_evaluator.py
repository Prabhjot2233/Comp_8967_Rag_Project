import pytest
from src.evaluator import RAGEvaluator

def test_mrr_calculation():
    # Setup retrieved and ground truth document IDs
    # Case 1: First retrieval is correct -> RR = 1.0
    # Case 2: Third retrieval is correct -> RR = 1.0/3.0
    # Case 3: No retrieval is correct -> RR = 0.0
    # Mean of [1.0, 0.333, 0.0] = 0.444
    
    retrieved = [
        ["doc1", "doc2", "doc3"],
        ["doc4", "doc5", "doc6"],
        ["doc7", "doc8", "doc9"]
    ]
    
    ground_truth = [
        ["doc1"],
        ["doc6"],
        ["doc10"]
    ]
    
    mrr = RAGEvaluator.compute_mrr(retrieved, ground_truth)
    expected_mrr = (1.0 + (1.0 / 3.0) + 0.0) / 3.0
    
    assert abs(mrr - expected_mrr) < 1e-6
    
    # Empty inputs should return 0.0
    assert RAGEvaluator.compute_mrr([], []) == 0.0

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.pipeline import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    @patch('src.pipeline.RAGEmbedder')
    @patch('src.pipeline.OllamaGenerator')
    def test_e2e_pipeline_flow(self, mock_generator_class, mock_embedder_class):
        # 1. Setup mock embedder
        mock_embedder = MagicMock()
        # Mock embed_texts to return numpy array of shape (2, 4)
        mock_embedder.embed_texts.return_value = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype='float32')
        # Mock embed_query to return a 1D vector
        mock_embedder.embed_query.return_value = np.array([0.0, 1.0, 0.0, 0.0], dtype='float32')
        mock_embedder_class.return_value = mock_embedder
        
        # 2. Setup mock generator
        mock_generator = MagicMock()
        mock_generator.generate_answer.return_value = "Test response"
        mock_generator_class.return_value = mock_generator
        
        # 3. Initialize Pipeline
        pipeline = RAGPipeline(embedder_model="mock-model", index_type="flat", llm_model="mock-llm")
        
        doc_ids = ["doc_A", "doc_B"]
        doc_texts = ["Context text A", "Context text B"]
        
        pipeline.build_system(doc_ids, doc_texts)
        
        # Run query
        result = pipeline.query("Query text", k=1)
        
        # Assertions
        self.assertEqual(result["query"], "Query text")
        self.assertEqual(result["retrieved_ids"], ["doc_B"])
        self.assertEqual(result["contexts"], ["Context text B"])
        self.assertEqual(result["answer"], "Test response")
        
        # Verify generator was called with query and retrieved context
        mock_generator.generate_answer.assert_called_with("Query text", ["Context text B"])

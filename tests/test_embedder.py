import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.embedder import RAGEmbedder

class TestEmbedder(unittest.TestCase):
    @patch('src.embedder.SentenceTransformer')
    def test_embedding_shape_and_e5_prefix(self, mock_transformer_class):
        # Setup mock
        mock_model = MagicMock()
        # Mock encode to return a dummy numpy array
        mock_model.encode.return_value = np.random.rand(2, 384)
        mock_transformer_class.return_value = mock_model
        
        # Test BGE-small (should not prefix sentences with 'passage:')
        embedder = RAGEmbedder(model_name="BAAI/bge-small-en-v1.5")
        texts = ["hello world", "rag systems are cool"]
        embeddings = embedder.embed_texts(texts)
        
        # Verify encode was called with unmodified texts
        mock_model.encode.assert_called_with(texts, convert_to_numpy=True, show_progress_bar=False)
        self.assertEqual(embeddings.shape, (2, 384))
        
        # Reset mock calls
        mock_model.reset_mock()
        
        # Test E5 (should prefix sentences with 'passage:')
        mock_model.encode.return_value = np.random.rand(2, 768)
        e5_embedder = RAGEmbedder(model_name="intfloat/e5-base-v2")
        e5_embeddings = e5_embedder.embed_texts(texts)
        
        # Verify encode was called with prefixed texts
        expected_prefixed_texts = ["passage: hello world", "passage: rag systems are cool"]
        mock_model.encode.assert_called_with(expected_prefixed_texts, convert_to_numpy=True, show_progress_bar=False)
        self.assertEqual(e5_embeddings.shape, (2, 768))

    @patch('src.embedder.SentenceTransformer')
    def test_query_embedding_prefix(self, mock_transformer_class):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 768)
        mock_transformer_class.return_value = mock_model
        
        e5_embedder = RAGEmbedder(model_name="intfloat/e5-base-v2")
        _ = e5_embedder.embed_query("what is RAG?")
        
        mock_model.encode.assert_called_with(["query: what is RAG?"], convert_to_numpy=True, show_progress_bar=False)

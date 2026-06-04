import unittest
from unittest.mock import MagicMock, patch
from src.generator import OllamaGenerator

class TestGenerator(unittest.TestCase):
    @patch('src.generator.ollama.Client')
    def test_ollama_generator(self, mock_client_class):
        # Setup mock
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "Mocked response from LLM"}
        mock_client_class.return_value = mock_client
        
        generator = OllamaGenerator(model_name="llama3")
        
        # Test prompt creation
        query = "what is the capital of France?"
        contexts = ["Paris is the capital of France.", "France is in Europe."]
        prompt = generator.generate_prompt(query, contexts)
        
        self.assertIn("Paris is the capital of France.", prompt)
        self.assertIn("what is the capital of France?", prompt)
        
        # Test answer generation
        answer = generator.generate_answer(query, contexts)
        self.assertEqual(answer, "Mocked response from LLM")
        
        # Verify mock calls
        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args[1]
        self.assertEqual(call_kwargs['model'], "llama3")
        self.assertEqual(call_kwargs['prompt'], prompt)

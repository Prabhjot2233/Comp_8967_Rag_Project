import ollama
from typing import List

class OllamaGenerator:
    def __init__(self, model_name: str = "llama3", host: str = "http://localhost:11434"):
        """
        Initializes the Ollama generator.
        """
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def generate_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Constructs a prompt with the user query and retrieved contexts.
        """
        context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        prompt = (
            f"You are a helpful assistant. Answer the question based ONLY on the following context. "
            f"If the answer cannot be found in the context, say 'I cannot find the answer in the provided context.'\n\n"
            f"--- Context ---\n{context_str}\n\n"
            f"--- Question ---\n{query}\n\n"
            f"--- Answer ---"
        )
        return prompt

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        Generates an answer from Ollama using the retrieved contexts.
        """
        prompt = self.generate_prompt(query, contexts)
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt
            )
            return response.get('response', '').strip()
        except Exception as e:
            return f"Error during generation: {str(e)}"

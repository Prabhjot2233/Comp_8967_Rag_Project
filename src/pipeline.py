from typing import List, Dict, Any
from src.embedder import RAGEmbedder
from src.indexer import FAISSIndexer
from src.generator import OllamaGenerator

class RAGPipeline:
    def __init__(self, embedder_model: str = "BAAI/bge-small-en-v1.5", 
                 index_type: str = "flat", 
                 llm_model: str = "llama3",
                 ollama_host: str = "http://localhost:11434"):
        """
        Coordinates the full RAG workflow.
        """
        self.embedder = RAGEmbedder(model_name=embedder_model)
        self.indexer = None
        self.index_type = index_type
        self.llm_model = llm_model
        self.ollama_host = ollama_host
        self.generator = OllamaGenerator(model_name=llm_model, host=ollama_host)
        
        # In-memory document storage mapping ID to text content
        self.doc_store = {}

    def build_system(self, doc_ids: List[str], doc_texts: List[str]):
        """
        Embeds documents and builds the FAISS index.
        """
        assert len(doc_ids) == len(doc_texts), "doc_ids and doc_texts must match in length"
        self.doc_store = dict(zip(doc_ids, doc_texts))
        
        # Generate embeddings
        embeddings = self.embedder.embed_texts(doc_texts)
        dimension = embeddings.shape[1] if len(embeddings) > 0 else 384
        
        # Initialize and build index
        self.indexer = FAISSIndexer(dimension=dimension, index_type=self.index_type)
        self.indexer.build_index(embeddings, doc_ids)

    def query(self, query_text: str, k: int = 3) -> Dict[str, Any]:
        """
        Runs a query through the full RAG pipeline.
        Returns the retrieved contexts and the final LLM response.
        """
        if self.indexer is None:
            raise ValueError("System has not been built/indexed yet.")
            
        # 1. Embed query
        query_vector = self.embedder.embed_query(query_text)
        query_vector = query_vector.reshape(1, -1)  # Reshape for FAISS search
        
        # 2. Search FAISS
        distances, retrieved_ids = self.indexer.search(query_vector, k=k)
        retrieved_ids = retrieved_ids[0]  # Single query search
        
        # 3. Fetch texts from doc_store
        contexts = [self.doc_store[doc_id] for doc_id in retrieved_ids if doc_id in self.doc_store]
        
        # 4. Generate answer via Ollama
        answer = self.generator.generate_answer(query_text, contexts)
        
        return {
            "query": query_text,
            "retrieved_ids": retrieved_ids,
            "contexts": contexts,
            "distances": distances[0].tolist(),
            "answer": answer
        }

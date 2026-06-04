import os
import pandas as pd

class MSMARCOReader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_queries(self, file_path: str) -> pd.DataFrame:
        """Loads queries from a TSV or JSON file."""
        # Skeleton implementation
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Queries file not found: {file_path}")
        return pd.read_csv(file_path, sep='\t', names=['query_id', 'query_text'])

    def load_passages(self, file_path: str) -> pd.DataFrame:
        """Loads passages from a TSV or JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Passages file not found: {file_path}")
        return pd.read_csv(file_path, sep='\t', names=['passage_id', 'passage_text'])

    def load_qrels(self, file_path: str) -> pd.DataFrame:
        """Loads relevance judgments (queries to passage mappings)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Qrels file not found: {file_path}")
        # MS MARCO qrels: query_id, iteration (always 0), passage_id, relevance (0 or 1)
        return pd.read_csv(file_path, sep='\t', names=['query_id', 'iteration', 'passage_id', 'relevance'])

import os
import tempfile
import pytest
import pandas as pd
from src.data_pipeline import MSMARCOReader

def test_data_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy TSV files
        queries_path = os.path.join(tmpdir, "queries.tsv")
        passages_path = os.path.join(tmpdir, "passages.tsv")
        qrels_path = os.path.join(tmpdir, "qrels.tsv")
        
        with open(queries_path, "w") as f:
            f.write("q1\twhat is rag?\nq2\thow does faiss work?\n")
            
        with open(passages_path, "w") as f:
            f.write("p1\trag stands for retrieval augmented generation\np2\tfaiss is a library for similarity search\n")
            
        with open(qrels_path, "w") as f:
            f.write("q1\t0\tp1\t1\nq2\t0\tp2\t1\n")
            
        reader = MSMARCOReader(data_dir=tmpdir)
        
        queries_df = reader.load_queries(queries_path)
        passages_df = reader.load_passages(passages_path)
        qrels_df = reader.load_qrels(qrels_path)
        
        assert len(queries_df) == 2
        assert queries_df.iloc[0]['query_id'] == 'q1'
        assert queries_df.iloc[0]['query_text'] == 'what is rag?'
        
        assert len(passages_df) == 2
        assert passages_df.iloc[1]['passage_id'] == 'p2'
        
        assert len(qrels_df) == 2
        assert qrels_df.iloc[0]['query_id'] == 'q1'
        assert qrels_df.iloc[0]['passage_id'] == 'p1'
        assert qrels_df.iloc[0]['relevance'] == 1

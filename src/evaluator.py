from typing import List, Dict

class RAGEvaluator:
    @staticmethod
    def compute_mrr(retrieved_ids: List[List[str]], ground_truth_ids: List[List[str]]) -> float:
        """
        Computes Mean Reciprocal Rank (MRR).
        For each query, retrieved_ids contains the ordered list of retrieved passage IDs.
        ground_truth_ids contains the list of true relevant passage IDs.
        """
        assert len(retrieved_ids) == len(ground_truth_ids), "Length mismatch between predictions and ground truths"
        
        rr_sum = 0.0
        for ret, gt in zip(retrieved_ids, ground_truth_ids):
            gt_set = set(gt)
            rank = 0
            for idx, item in enumerate(ret):
                if item in gt_set:
                    rank = idx + 1
                    break
            if rank > 0:
                rr_sum += 1.0 / rank
                
        return rr_sum / len(retrieved_ids) if retrieved_ids else 0.0

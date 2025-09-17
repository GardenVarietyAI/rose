import logging
from pathlib import Path
from typing import List

import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        # Initialize ONNX session
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        options.inter_op_num_threads = 4
        options.intra_op_num_threads = 4

        model_file = self.model_path / "model.onnx"
        if not model_file.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_file}")

        self.session = InferenceSession(str(model_file), options, providers=["CPUExecutionProvider"])
        logger.info(f"Loaded ONNX model from {model_file}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        logger.info(f"Loaded tokenizer with {len(self.tokenizer)} tokens")

    def score(self, query: str, response: str, max_length: int = 2048) -> float:
        inputs = self.tokenizer(
            f"Query: {query}\nResponse: {response}",
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )

        # Get logits and convert to probability
        logits = outputs[0][0]

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        # Return relevance score (index 0 for Qwen reranker)
        return float(probabilities[0])

    def score_batch(
        self,
        queries: List[str],
        responses: List[str],
        max_length: int = 2048,
    ) -> List[float]:
        if len(queries) != len(responses):
            raise ValueError("Queries and responses must have same length")

        scores = []
        for query, response in zip(queries, responses):
            scores.append(self.score(query, response, max_length))

        return scores

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.session: Optional[InferenceSession] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.config = self._load_config()
        self._initialize_model()

    def _load_config(self) -> Dict[str, Any]:
        config_path = self.model_path / "reranker_config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {
            "instruction_template": "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}",
            "default_instruction": "Given a web search query, retrieve relevant passages that answer the query",
            "max_length": 2048,
        }

    def _initialize_model(self) -> None:
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        options.inter_op_num_threads = 4
        options.intra_op_num_threads = 4

        # Load ONNX model
        model_file = self.model_path / "model.onnx"
        if not model_file.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_file}")

        self.session = InferenceSession(
            str(model_file),
            options,
            providers=["CPUExecutionProvider"],  # Skip CoreML due to embedding size
        )
        logger.info(f"Loaded ONNX model from {model_file}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        logger.info(f"Loaded tokenizer with {len(self.tokenizer)} tokens")

    def format_input(
        self,
        query: str,
        document: str,
        instruction: Optional[str] = None,
    ) -> str:
        if instruction is None:
            instruction = self.config["default_instruction"]

        template = self.config["instruction_template"]
        return template.format(
            instruction=instruction,
            query=query,
            document=document,
        )

    def score(
        self,
        query: str,
        response: str,
        instruction: Optional[str] = None,
    ) -> float:
        if not self.session or not self.tokenizer:
            raise RuntimeError("Model not initialized")

        # Format input
        input_text = self.format_input(query, response, instruction)

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.config.get("max_length", 2048),
        )

        # Run inference
        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )

        # Get logits and convert to probability
        logits = outputs[0][0]  # Shape: [batch_size, num_labels]

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        # Qwen reranker uses index 0 as the relevance score
        # Higher score = more relevant/better response
        return float(probabilities[0])

    def score_batch(
        self,
        queries: List[str],
        responses: List[str],
        instruction: Optional[str] = None,
    ) -> List[float]:
        if len(queries) != len(responses):
            raise ValueError("Queries and responses must have same length")

        scores = []
        for query, response in zip(queries, responses):
            scores.append(self.score(query, response, instruction))

        return scores

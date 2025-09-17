import logging
from typing import List

import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def score(
    query: str, response: str, session: InferenceSession, tokenizer: AutoTokenizer, max_length: int = 2048
) -> float:
    inputs = tokenizer(
        f"Query: {query}\nResponse: {response}",
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
    )

    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    return float(probabilities[0])


def score_batch(
    queries: List[str],
    responses: List[str],
    session: InferenceSession,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
) -> List[float]:
    if len(queries) != len(responses):
        raise ValueError("Queries and responses must have same length")

    scores = []
    for query, response in zip(queries, responses):
        scores.append(score(query, response, session, tokenizer, max_length))

    return scores

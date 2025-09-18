import logging
from typing import List

import numpy as np
from onnxruntime import InferenceSession
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


def score(query: str, response: str, session: InferenceSession, tokenizer: Tokenizer, max_length: int = 512) -> float:
    # MS-Marco expects [CLS] query [SEP] document [SEP]
    encoding = tokenizer.encode(query, response)

    if len(encoding.ids) > max_length:
        encoding.truncate(max_length)

    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )

    # MS-Marco outputs a single relevance logit
    logit = float(outputs[0][0])

    # Apply sigmoid to get probability
    relevance_score = float(1 / (1 + np.exp(-logit)))

    logger.debug(f"Query: {query[:50]}, Doc: {response[:50]}, Logit: {logit:.2f}, Score: {relevance_score:.4f}")

    return relevance_score


def score_batch(
    queries: List[str],
    responses: List[str],
    session: InferenceSession,
    tokenizer: Tokenizer,
    max_length: int = 512,
) -> List[float]:
    if len(queries) != len(responses):
        raise ValueError("Queries and responses must have same length")

    if len(queries) == 0:
        return []

    # Encode all pairs
    encodings = []
    for query, response in zip(queries, responses):
        encoding = tokenizer.encode(query, response)
        if len(encoding.ids) > max_length:
            encoding.truncate(max_length)
        encodings.append(encoding)

    # Find max length for padding
    max_len = max(len(enc.ids) for enc in encodings)

    # Create batched arrays with padding
    batch_size = len(encodings)
    input_ids = np.zeros((batch_size, max_len), dtype=np.int64)
    attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)

    for i, encoding in enumerate(encodings):
        seq_len = len(encoding.ids)
        input_ids[i, :seq_len] = encoding.ids
        attention_mask[i, :seq_len] = encoding.attention_mask

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )

    # Extract logits and apply sigmoid
    logits = outputs[0].squeeze(axis=-1)  # Shape: [batch_size]
    relevance_scores = 1 / (1 + np.exp(-logits))
    scores: List[float] = relevance_scores.tolist()
    return scores

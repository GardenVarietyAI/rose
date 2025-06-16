"""Common utilities for scoring functions.
normalize_answer function adapted from:
https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
MIT License
Copyright (c) 2016 Pranav Rajpurkar, Stanford NLP
"""
import re
import string


def normalize_answer(s):
    """Normalize answer string for comparison.

    Adapted from the official SQuAD evaluation script.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
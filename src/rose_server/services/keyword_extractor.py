import re
from collections import Counter
from dataclasses import dataclass

from rose_server.services.stopwords import EN_STOPWORDS, EN_WHITELIST


@dataclass
class KeywordExtractionResult:
    phrases: list[str]
    tokens: list[str]
    scores: dict[str, float]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w.\-]+", text.lower())


def _iter_keyword_phrases(tokens: list[str], stopwords_set: set[str], whitelist: frozenset[str]) -> list[str]:
    """Extract keyword phrases from tokens."""
    if not tokens:
        return []

    phrases: list[str] = []
    phrase_tokens: list[str] = []

    for token in tokens:
        if ("." in token or "_" in token or "-" in token) and len(token) >= 3:
            phrase_tokens.append(token)
            continue

        if token in whitelist:
            phrase_tokens.append(token)
            continue

        if len(token) < 3 or token in stopwords_set:
            if phrase_tokens:
                phrases.append(" ".join(phrase_tokens))
                phrase_tokens = []
            continue

        phrase_tokens.append(token)

    if phrase_tokens:
        phrases.append(" ".join(phrase_tokens))

    return phrases


def _score_phrases(phrases: list[str]) -> list[tuple[float, str]]:
    """Score phrases by word frequency and length."""
    word_counts = Counter(word for phrase in phrases for word in phrase.split())

    scored: list[tuple[float, str]] = []
    for phrase in phrases:
        words = phrase.split()
        score = float(len(words) * sum(word_counts[word] for word in words))
        scored.append((score, phrase))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def extract_keywords(
    text: str,
    extra_stopwords: set[str] | None = None,
    max_keywords: int = 5,
) -> KeywordExtractionResult:
    """Extract keywords and phrases from text."""
    tokens = tokenize(text)

    if extra_stopwords:
        stopwords = EN_STOPWORDS | extra_stopwords
    else:
        stopwords = EN_STOPWORDS

    phrases = _iter_keyword_phrases(tokens, stopwords, EN_WHITELIST)
    if not phrases:
        return KeywordExtractionResult(phrases=[], tokens=tokens, scores={})

    scored = _score_phrases(phrases)

    top_phrases: list[str] = []
    seen: set[str] = set()
    scores_dict: dict[str, float] = {}

    for score, phrase in scored:
        if phrase in seen:
            continue
        seen.add(phrase)
        scores_dict[phrase] = score
        top_phrases.append(phrase)
        if len(top_phrases) >= max_keywords:
            break

    return KeywordExtractionResult(
        phrases=top_phrases,
        tokens=tokens,
        scores=scores_dict,
    )

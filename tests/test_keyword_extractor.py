from rose_server.services.keyword_extractor import extract_keywords


def test_extract_keywords_all_stopwords_returns_empty() -> None:
    result = extract_keywords("the and of")
    assert result.phrases == []


def test_extract_keywords_keeps_whitelisted_two_char_tokens() -> None:
    result = extract_keywords("db ui js")
    assert result.phrases
    phrase_tokens = set(result.phrases[0].split())
    assert {"db", "ui", "js"}.issubset(phrase_tokens)


def test_extract_keywords_position_boost_prefers_first_phrase() -> None:
    result = extract_keywords("alpha beta and gamma delta", max_keywords=1)
    assert result.phrases == ["alpha beta"]


def test_extract_keywords_scores_dict_matches_phrases() -> None:
    result = extract_keywords("alpha beta and gamma delta", max_keywords=2)
    assert result.phrases
    assert set(result.scores.keys()) == set(result.phrases)
    assert all(score > 0 for score in result.scores.values())


def test_extract_keywords_dedupes_by_normalized_key() -> None:
    result = extract_keywords("alpha beta and alpha beta", max_keywords=10)
    assert result.phrases == ["alpha beta"]
    assert list(result.scores.keys()) == ["alpha beta"]


def test_extract_keywords_preserves_technical_tokens() -> None:
    result = extract_keywords("process_data using file.txt and dash-case")
    assert "process_data" in result.tokens
    assert "file.txt" in result.tokens
    assert "dash-case" in result.tokens


def test_extract_keywords_respects_extra_stopwords() -> None:
    result = extract_keywords("alpha beta gamma", extra_stopwords={"beta"})
    phrase_tokens = set(word for phrase in result.phrases for word in phrase.split())
    assert "alpha" in phrase_tokens
    assert "gamma" in phrase_tokens
    assert "beta" not in phrase_tokens


def test_extract_keywords_returns_all_tokens() -> None:
    result = extract_keywords("alpha beta and gamma")
    assert result.tokens == ["alpha", "beta", "and", "gamma"]

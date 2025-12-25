import re


def parse_query(text: str) -> tuple[str, str, list[str]]:
    """Extract @mentions from query."""
    pattern = r"(?<!\w)@(\w+)"
    lens_at_names = [name.lower() for name in re.findall(pattern, text)]
    clean_query = re.sub(pattern, "", text)
    clean_query = " ".join(clean_query.split())

    return text, clean_query, lens_at_names

import sys
sys.path.append('src')

from rose_server import utils


def test_extract_user_content_returns_none_for_invalid_type():
    assert utils.extract_user_content(123) is None


def test_extract_user_content_returns_none_when_no_input_text():
    content = [{"type": "image", "url": "image.png"}]
    assert utils.extract_user_content(content) is None


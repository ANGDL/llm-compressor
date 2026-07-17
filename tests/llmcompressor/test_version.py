import pytest

from llmcompressor._version import _normalize_tag


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("k-v0.1.0", "0.1.0"),
        ("v0.12.0", "0.12.0"),
        ("0.12.0", "0.12.0"),
    ],
)
def test_normalize_tag(tag: str, expected: str):
    assert _normalize_tag(tag) == expected

from llmcompressor._version import _COMMIT_RE, _to_version_tuple, _with_commit


def test_version_appends_commit_when_metadata_has_no_revision():
    assert _with_commit("0.15.0", "5ccf2a1dab3e", False) == (
        "0.15.0+g5ccf2a1dab3e"
    )


def test_version_keeps_existing_revision():
    version = "0.15.0.dev4+g5ccf2a1dab3e"
    assert _with_commit(version, "5ccf2a1dab3e", False) == version
    assert _COMMIT_RE.search(version)


def test_version_marks_dirty_working_tree():
    assert _with_commit("0.15.0", "5ccf2a1dab3e", True) == (
        "0.15.0+g5ccf2a1dab3e.dirty"
    )


def test_version_tuple_keeps_commit_segment():
    assert _to_version_tuple("0.15.0+g5ccf2a1dab3e") == (
        0,
        15,
        0,
        "g5ccf2a1dab3e",
    )

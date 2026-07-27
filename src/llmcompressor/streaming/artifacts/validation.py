"""Compatibility checks for resuming streaming compression artifacts."""

from __future__ import annotations

from dataclasses import fields

from .manifest import StreamingRunManifest


class ArtifactCompatibilityError(ValueError):
    """Raised when an artifact belongs to a different streaming run."""


def validate_manifest_compatibility(
    stored: StreamingRunManifest, expected: StreamingRunManifest
) -> None:
    """Reject reuse when any result-affecting manifest section changed."""

    mismatches = []
    for field in fields(StreamingRunManifest):
        name = field.name
        stored_value = getattr(stored, name)
        expected_value = getattr(expected, name)
        if name == "source":
            stored_value = stored.source.content_fingerprint
            expected_value = expected.source.content_fingerprint
        elif name == "software":
            # Software versions are recorded for diagnostics. They are not part of
            # result identity because a resumed stage may run in an upgraded but
            # compatible environment. Schema changes handle incompatible formats.
            continue
        if stored_value != expected_value:
            mismatches.append(name)

    if mismatches:
        raise ArtifactCompatibilityError(
            "Streaming artifact is incompatible with the requested run; "
            f"changed sections: {', '.join(mismatches)}"
        )

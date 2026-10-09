"""Static browser UI assets served by the dependency-free WSGI app."""

from __future__ import annotations

from pathlib import Path

from .store import NotFoundError

_UI_ROOT = Path(__file__).with_name("ui")
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/ui": ("index.html", "text/html; charset=utf-8"),
    "/ui/index.html": ("index.html", "text/html; charset=utf-8"),
    "/ui/app.css": ("app.css", "text/css; charset=utf-8"),
    "/ui/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def asset(path: str) -> tuple[bytes, str, list[tuple[str, str]]]:
    """Return an allowlisted UI asset and its security headers."""

    record = _ASSETS.get(path)
    if record is None:
        raise NotFoundError(f"unknown UI asset: {path}")
    filename, content_type = record
    try:
        body = (_UI_ROOT / filename).read_bytes()
    except OSError as error:
        raise NotFoundError(f"UI asset {filename!r} is unavailable") from error
    headers = [
        (
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'",
        ),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "same-origin"),
    ]
    return body, content_type, headers


def is_ui_path(path: str) -> bool:
    return path == "/" or path == "/ui" or path.startswith("/ui/")

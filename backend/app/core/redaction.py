"""Secret redaction for logs and error messages.

Redacts:
- credentials embedded in database URLs
- literal secret values supplied by configuration
- key=value / key: value pairs for common secret names
- bearer tokens
- Google (AIza…), Groq (gsk_…), and Tavily (tvly-…) key formats
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import unquote, urlparse, urlunparse

REDACTED = "[redacted]"

_URL_PATTERN = re.compile(
    r"postgres(?:ql)?(?:\+asyncpg)?://[^\s\"'<>]+",
    re.IGNORECASE,
)
_KEY_VALUE_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password)\b"
    r"(\s*[=:]\s*)([\"']?)([^\s\"',;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+")
_TOKEN_PATTERNS = (
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bgsk_[0-9A-Za-z]{20,}\b"),
    re.compile(r"\btvly-[0-9A-Za-z_-]{10,}\b"),
)


def redact_url(url: str) -> str:
    """Remove username and password from a URL while keeping host/port/path."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return REDACTED

    if not parsed.username and not parsed.password:
        return url

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Return ``text`` with known secrets and secret-shaped values removed."""
    if not text:
        return text

    result = text

    # Literal configured secrets first (longest first, to avoid partial leaks).
    for secret in sorted({value for value in secrets if value}, key=len, reverse=True):
        result = result.replace(secret, REDACTED)
        decoded = unquote(secret)
        if decoded != secret:
            result = result.replace(decoded, REDACTED)

    result = _URL_PATTERN.sub(lambda match: redact_url(match.group(0)), result)
    result = _KEY_VALUE_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}",
        result,
    )
    result = _BEARER_PATTERN.sub(lambda match: f"{match.group(1)} {REDACTED}", result)
    for pattern in _TOKEN_PATTERNS:
        result = pattern.sub(REDACTED, result)
    return result

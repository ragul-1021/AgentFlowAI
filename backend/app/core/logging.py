"""Structured JSON logging with secret redaction."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterable
from datetime import datetime, timezone

from app.core.redaction import redact

_EXTRA_FIELDS = ("request_id", "environment", "mock_mode", "stage", "task_id")
_configured = False


class JsonFormatter(logging.Formatter):
    """One JSON object per log line, with secrets removed."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(value for value in secrets if value)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, ensure_ascii=False), self._secrets)


class RedactionFilter(logging.Filter):
    """Redact log messages and exception text before any handler sees them."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(value for value in secrets if value)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive for malformed records
            return True
        record.msg = redact(message, self._secrets)
        record.args = ()
        if record.exc_info:
            record.exc_text = None
        return True


def configure_logging(level: str = "INFO", secrets: Iterable[str] = ()) -> None:
    """Configure root and uvicorn logging once per process."""
    global _configured
    if _configured:
        return

    secret_values = tuple(value for value in secrets if value)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(secret_values))
    handler.addFilter(RedactionFilter(secret_values))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    _configured = True

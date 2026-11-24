"""Pydantic request/response schemas (Phase 2 core)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.enums import ReportFormat, ReportLength, ResearchDepth

_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)

_DEPTH_SOURCE_DEFAULTS = {
    ResearchDepth.QUICK: 8,
    ResearchDepth.STANDARD: 12,
    ResearchDepth.DEEP: 18,
}


def _normalize_domain(value: str) -> str:
    domain = value.strip().lower().rstrip(".")
    if "://" in domain or "/" in domain or " " in domain:
        raise ValueError("domains must be bare hostnames such as example.com")
    if not _HOSTNAME_RE.match(domain):
        raise ValueError(f"invalid domain: {value!r}")
    return domain


class TaskCreate(BaseModel):
    """Validated payload for creating a research task."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: str = Field(min_length=10, max_length=500)
    description: str = Field(default="", max_length=4_000)
    depth: ResearchDepth = ResearchDepth.STANDARD
    source_target: int | None = Field(default=None, ge=5, le=30)
    format: ReportFormat = ReportFormat.GENERAL
    length: ReportLength = ReportLength.STANDARD
    preferred_domains: list[str] = Field(default_factory=list, max_length=10)
    excluded_domains: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("preferred_domains", "excluded_domains")
    @classmethod
    def _validate_domains(cls, values: list[str]) -> list[str]:
        normalized = [_normalize_domain(value) for value in values]
        deduped = list(dict.fromkeys(normalized))
        if len(deduped) > 10:
            raise ValueError("at most 10 domains are allowed per list")
        return deduped

    @model_validator(mode="after")
    def _validate_domain_lists(self) -> "TaskCreate":
        overlap = set(self.preferred_domains) & set(self.excluded_domains)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                f"preferred_domains and excluded_domains must not overlap: {names}"
            )
        return self

    @property
    def resolved_source_target(self) -> int:
        """Explicit source count wins; otherwise use the depth default."""
        if self.source_target is not None:
            return self.source_target
        return _DEPTH_SOURCE_DEFAULTS[self.depth]

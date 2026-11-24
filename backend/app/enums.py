"""Domain enums shared by models, schemas, and the pipeline.

All enum-like database columns are stored as ``varchar`` with a CHECK
constraint (no native PostgreSQL enum types), so adding a value is a normal
migration rather than an ``ALTER TYPE``.
"""

from __future__ import annotations

from enum import StrEnum


class TaskStatus(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    WRITING = "writing"
    REVIEWING = "reviewing"
    REVISING = "revising"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStage(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    WRITING = "writing"
    REVIEWING = "reviewing"
    REVISING = "revising"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchDepth(StrEnum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ReportFormat(StrEnum):
    GENERAL = "general"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    MARKET_RESEARCH = "market_research"
    COMPARATIVE = "comparative"
    TECHNOLOGY = "technology"


class ReportLength(StrEnum):
    SHORT = "short"
    STANDARD = "standard"
    LONG = "long"


class ReportStatus(StrEnum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class CitationVerdict(StrEnum):
    UNCHECKED = "unchecked"
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class ExecutionStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    RETRYING = "retrying"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentName(StrEnum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"
    SYSTEM = "system"


class FindingCategory(StrEnum):
    FINDING = "finding"
    PATTERN = "pattern"
    CONTRADICTION = "contradiction"
    GAP = "gap"
    LIMITATION = "limitation"


class EvidenceStrength(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SourceStatus(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


class EventType(StrEnum):
    SNAPSHOT = "snapshot"
    STAGE = "stage"
    AGENT = "agent"
    LOG = "log"
    SOURCE = "source"
    REPORT = "report"
    ERROR = "error"
    PING = "ping"
    DONE = "done"


def values(enum_cls: type[StrEnum]) -> list[str]:
    """Return the string values of an enum for CHECK constraints."""
    return [member.value for member in enum_cls]

"""Tests for PipelineState transitions, telemetry, and cancellation."""

from __future__ import annotations

from decimal import Decimal
import uuid
import pytest

from app.enums import TaskStage, TaskStatus
from app.orchestration import PipelineCancelledError, PipelineState


def test_pipeline_state_initialization():
    task_id = uuid.uuid4()
    state = PipelineState(task_id=task_id)

    assert state.task_id == task_id
    assert state.current_stage == TaskStage.QUEUED
    assert state.task_status == TaskStatus.QUEUED
    assert state.progress == 0
    assert state.total_tokens == 0
    assert state.total_calls == 0
    assert state.estimated_cost == Decimal("0.000000")
    assert state.cancellation_requested is False


def test_pipeline_state_transitions():
    state = PipelineState(task_id=uuid.uuid4())

    state.transition_to(TaskStage.PLANNING, TaskStatus.PLANNING, progress=15)
    assert state.current_stage == TaskStage.PLANNING
    assert state.task_status == TaskStatus.PLANNING
    assert state.progress == 15

    # Progress is clamped between 0 and 100
    state.transition_to(TaskStage.COMPLETED, TaskStatus.PENDING_APPROVAL, progress=150)
    assert state.progress == 100

    state.transition_to(TaskStage.FAILED, TaskStatus.FAILED, progress=-20)
    assert state.progress == 0


def test_pipeline_state_usage_accumulation():
    state = PipelineState(task_id=uuid.uuid4())
    state.accumulate_usage(100, 50, Decimal("0.000150"))
    assert state.total_tokens == 150
    assert state.total_calls == 1
    assert state.estimated_cost == Decimal("0.000150")

    state.accumulate_usage(200, 100, Decimal("0.000300"))
    assert state.total_tokens == 450
    assert state.total_calls == 2
    assert state.estimated_cost == Decimal("0.000450")


def test_pipeline_cancellation_request_and_check():
    state = PipelineState(task_id=uuid.uuid4())
    state.transition_to(TaskStage.RESEARCHING, TaskStatus.RESEARCHING)

    # Initial check does not raise
    state.check_cancellation()

    # Request cancellation
    state.request_cancellation()
    assert state.cancellation_requested is True

    # Subsequent check raises PipelineCancelledError
    with pytest.raises(PipelineCancelledError) as exc_info:
        state.check_cancellation()

    assert "was cancelled" in str(exc_info.value)
    assert exc_info.value.stage == "researching"

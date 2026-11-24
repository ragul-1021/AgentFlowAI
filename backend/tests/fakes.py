"""Test doubles shared across suites."""

from __future__ import annotations

from app.db import DatabaseProbe


class FakeDatabase:
    """Deterministic database stand-in for HTTP tests."""

    def __init__(
        self,
        status: str = "up",
        revision: str | None = "0001",
        latency_ms: int = 1,
    ) -> None:
        self.status = status
        self.revision = revision
        self.latency_ms = latency_ms
        self.disposed = False

    async def probe(self) -> DatabaseProbe:
        return DatabaseProbe(
            status=self.status,
            revision=self.revision if self.status == "up" else None,
            latency_ms=self.latency_ms,
            error_code=None if self.status == "up" else "unavailable",
        )

    async def dispose(self) -> None:
        self.disposed = True


class FakeSession:
    """In-memory async session double for offline pipeline tests."""

    def __init__(
        self,
        store: dict[tuple[type, Any], Any],
        history: list[Any],
    ) -> None:
        self._store = store
        self._history = history
        self.added: list[Any] = []
        self.committed = False
        self.rolled_back = False

    def add(self, obj: Any) -> None:
        self.added.append(obj)
        self._history.append(obj)
        if hasattr(obj, "id") and obj.id is not None:
            self._store[(type(obj), obj.id)] = obj

    async def get(self, model_cls: type, obj_id: Any) -> Any | None:
        return self._store.get((model_cls, obj_id))

    async def execute(self, stmt: Any) -> Any:
        class FakeResult:
            def __init__(self, items: list[Any]) -> None:
                self._items = items

            def scalars(self) -> Any:
                return self

            def first(self) -> Any | None:
                return self._items[0] if self._items else None

            def all(self) -> list[Any]:
                return self._items

        # Determine target model class from statement
        target_cls = None
        for entity in stmt._propagate_attrs.get("compile_state_plugin", {}):
            pass
        if hasattr(stmt, "columns_plus_names"):
            pass

        # Match against store by type
        from app.models import ResearchReport, RunEvent
        if "research_reports" in str(stmt):
            reports = [obj for obj in self._store.values() if isinstance(obj, ResearchReport)]
            return FakeResult(reports)
        elif "run_events" in str(stmt):
            events = [obj for obj in self._history if isinstance(obj, RunEvent)]
            last_id = 0
            if hasattr(stmt, "whereclause") and stmt.whereclause is not None:
                # Find integer comparisons in where clause
                for child in getattr(stmt.whereclause, "clauses", [stmt.whereclause]):
                    if hasattr(child, "right") and hasattr(child.right, "value"):
                        if isinstance(child.right.value, int):
                            last_id = max(last_id, child.right.value)
            filtered = [ev for ev in events if ev.id > last_id]
            return FakeResult(filtered)

        return FakeResult(list(self._store.values()))

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            await self.rollback()
        else:
            await self.commit()
        await self.close()


class FakeSessionmaker:
    """Factory yielding FakeSession instances sharing an in-memory object store."""

    def __init__(self) -> None:
        self.store: dict[tuple[type, Any], Any] = {}
        self.history: list[Any] = []

    def __call__(self) -> FakeSession:
        return FakeSession(self.store, self.history)

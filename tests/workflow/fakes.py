"""A scripted ``WorkflowModel`` for the Plan loop: replies queue per schema, and every call is recorded."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

Reply = BaseModel | Callable[[str], BaseModel]


@dataclass
class Call:
    schema: str
    system: str
    prompt: str


@dataclass
class SchemaModel:
    """Replies are consumed per schema name in call order; the last one repeats when ``repeat`` is set.
    A reply may be a callable of the prompt."""

    replies: dict[str, list[Reply]]
    calls: list[Call] = field(default_factory=list)
    repeat: bool = True

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        self.calls.append(Call(schema.__name__, system, prompt))
        queue = self.replies.get(schema.__name__)
        assert queue, f"no scripted reply for {schema.__name__}; prompt was:\n{prompt}"
        reply = queue.pop(0) if len(queue) > 1 or not self.repeat else queue[0]
        out: Any = reply(prompt) if callable(reply) else reply
        assert isinstance(out, schema)
        return out

    def of(self, schema: str) -> list[Call]:
        return [c for c in self.calls if c.schema == schema]

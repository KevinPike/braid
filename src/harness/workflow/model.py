"""Structured model calls: one schema-shaped JSON answer per call, through the guard client (ADR 0004)."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from harness.guard.ollama_client import OllamaClient

T = TypeVar("T", bound=BaseModel)


class WorkflowModelError(RuntimeError):
    """The model did not return a valid reply for the schema, even after one retry."""


class WorkflowModel(Protocol):
    async def complete(self, system: str, prompt: str, schema: type[T]) -> T: ...


class GuardedWorkflowModel:
    """Asks ``OllamaClient.chat`` for JSON in ``schema``'s shape, so ``num_ctx`` stays pinned for every request."""

    def __init__(self, client: OllamaClient, model_id: str, *, temperature: float = 0.2) -> None:
        self.client = client
        self.model_id = model_id
        self.temperature = temperature

    async def complete(self, system: str, prompt: str, schema: type[T]) -> T:
        error = ""
        for _ in range(2):  # one retry, with the validation error appended
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt + error}]
            text = "".join([
                chunk.text async for chunk in self.client.chat(
                    self.model_id, messages, format=schema.model_json_schema(), options={"temperature": self.temperature},
                )
            ])  # thinking text is in chunk.thinking and is dropped
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                error = (
                    f"\n\nYour last reply was invalid ({exc.error_count()} errors: {str(exc)[:300]}). "
                    "Reply with valid JSON only."
                )
        raise WorkflowModelError(f"{self.model_id} returned invalid JSON twice for {schema.__name__}:{error}")

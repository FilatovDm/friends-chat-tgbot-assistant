from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from openai import APITimeoutError, AsyncOpenAI

from database import ThreadDatabase

logger = logging.getLogger(__name__)


class AssistantRunTimeoutError(Exception):
    """Raised when a run does not finish within timeout."""


class AssistantRunFailedError(Exception):
    """Raised when a run finishes unsuccessfully."""


@dataclass(slots=True)
class AssistantConfig:
    assistant_id: str
    vector_store_id: str | None = None
    temperature: float = 1.0
    run_timeout_seconds: float = 120.0
    poll_interval_seconds: float = 1.0


class OpenAIAssistantService:
    def __init__(self, api_key: str, config: AssistantConfig) -> None:
        self._client = AsyncOpenAI(api_key=api_key, timeout=60.0)
        self._config = config

    async def configure_assistant(self) -> None:
        """Best effort: ensure assistant temperature/tools/tool_resources are set."""
        payload: dict[str, object] = {
            "temperature": self._config.temperature,
            "tools": [{"type": "file_search"}],
        }
        if self._config.vector_store_id:
            payload["tool_resources"] = {
                "file_search": {"vector_store_ids": [self._config.vector_store_id]}
            }

        try:
            await self._client.beta.assistants.update(
                assistant_id=self._config.assistant_id,
                **payload,
            )
            logger.info("Assistant configuration synced successfully.")
        except Exception:
            # Some accounts/features may not allow every tool type.
            # We keep running with existing assistant config instead of hard failing.
            logger.exception(
                "Could not sync assistant settings; continuing with existing configuration."
            )

    async def ask(
        self,
        chat_id: int,
        user_text: str,
        db: ThreadDatabase,
    ) -> str:
        thread_id = await self._get_or_create_thread(chat_id, db)

        await self._client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_text,
        )

        run = await self._client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self._config.assistant_id,
            temperature=self._config.temperature,
        )

        await self._wait_until_run_done(thread_id=thread_id, run_id=run.id)
        return await self._get_run_response(thread_id=thread_id, run_id=run.id)

    async def _get_or_create_thread(self, chat_id: int, db: ThreadDatabase) -> str:
        existing = db.get_thread_id(chat_id)
        if existing:
            return existing

        kwargs: dict[str, object] = {}
        if self._config.vector_store_id:
            kwargs["tool_resources"] = {
                "file_search": {"vector_store_ids": [self._config.vector_store_id]}
            }

        thread = await self._client.beta.threads.create(**kwargs)
        db.set_thread_id(chat_id, thread.id)
        logger.info("Created thread %s for chat %s", thread.id, chat_id)
        return thread.id

    async def _wait_until_run_done(self, thread_id: str, run_id: str) -> None:
        start = time.monotonic()

        while True:
            if time.monotonic() - start > self._config.run_timeout_seconds:
                raise AssistantRunTimeoutError(
                    f"Run {run_id} exceeded {self._config.run_timeout_seconds:.0f}s timeout."
                )

            run = await self._client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id,
            )
            status = run.status

            if status == "completed":
                return

            if status in {"failed", "cancelled", "expired"}:
                message = f"Run {run_id} ended with status={status}"
                if getattr(run, "last_error", None):
                    message = f"{message}. last_error={run.last_error}"
                raise AssistantRunFailedError(message)

            if status == "requires_action":
                raise AssistantRunFailedError(
                    "Run requires_action but no function-calling handler is configured."
                )

            await asyncio.sleep(self._config.poll_interval_seconds)

    async def _get_run_response(self, thread_id: str, run_id: str) -> str:
        messages = await self._client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc",
            limit=20,
        )

        for message in messages.data:
            if message.role != "assistant":
                continue
            if getattr(message, "run_id", None) != run_id:
                continue

            text_parts: list[str] = []
            for block in message.content:
                if block.type == "text":
                    text_parts.append(block.text.value)

            if text_parts:
                return "\n".join(text_parts).strip()

        return "не могу сейчас нормально ответить, попробуй позже."


__all__ = [
    "APITimeoutError",
    "AssistantConfig",
    "AssistantRunFailedError",
    "AssistantRunTimeoutError",
    "OpenAIAssistantService",
]

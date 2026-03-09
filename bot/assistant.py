from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from duckduckgo_search import DDGS
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
            "tools": [
                {"type": "file_search"},
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": (
                            "Search the internet for current information: "
                            "weather, news, facts, prices, anything requiring up-to-date data"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query in Russian or English",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                },
            ],
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
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for tc in tool_calls:
                    if tc.function.name == "web_search":
                        args = json.loads(tc.function.arguments)
                        result = await _run_web_search(args["query"])
                        tool_outputs.append(
                            {
                                "tool_call_id": tc.id,
                                "output": result,
                            }
                        )
                await self._client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run_id,
                    tool_outputs=tool_outputs,
                )
                continue

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

        return "что-то пошло не так, повтори позже."


async def _run_web_search(query: str) -> str:
    loop = asyncio.get_event_loop()

    def search() -> str:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return "Ничего не нашёл."

        parts: list[str] = []
        for r in results:
            parts.append(f"{r['title']}\n{r['body']}\n{r['href']}")
        return "\n\n".join(parts)

    return await loop.run_in_executor(None, search)


__all__ = [
    "APITimeoutError",
    "AssistantConfig",
    "AssistantRunFailedError",
    "AssistantRunTimeoutError",
    "OpenAIAssistantService",
]

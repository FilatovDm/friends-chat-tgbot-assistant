from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from collections.abc import Awaitable
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

from datetime import datetime
from dotenv import load_dotenv
from telegram import MessageEntity, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from assistant import (
    APITimeoutError,
    AssistantConfig,
    AssistantRunFailedError,
    AssistantRunTimeoutError,
    OpenAIAssistantService,
)
from database import ThreadDatabase


BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "bot.log"
DB_PATH = BASE_DIR / "threads.sqlite3"

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _is_mentioned(message_text: str, entities: list[MessageEntity], bot_username: str) -> bool:
    target = f"@{bot_username.lower()}"
    for entity in entities:
        if entity.type == MessageEntity.MENTION:
            mention = message_text[entity.offset : entity.offset + entity.length]
            if mention.lower() == target:
                return True
    return False


def _is_reply_to_bot(update: Update, bot_id: int) -> bool:
    message = update.effective_message
    if not message or not message.reply_to_message or not message.reply_to_message.from_user:
        return False
    return message.reply_to_message.from_user.id == bot_id


def should_respond(update: Update, bot_username: str, bot_id: int) -> bool:
    message = update.effective_message
    if not message or not message.text:
        return False

    if _is_reply_to_bot(update, bot_id):
        return True

    entities = message.entities or []
    if not entities or not bot_username:
        return False
    return _is_mentioned(message.text, entities, bot_username)


def strip_bot_mention(text: str, bot_username: str) -> str:
    if not bot_username:
        return text.strip()
    pattern = re.compile(rf"@{re.escape(bot_username)}", re.IGNORECASE)
    return pattern.sub("", text).strip()


def _parse_news_chat_ids(raw: str | None) -> list[int]:
    if not raw:
        return []
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            logger.warning("Не могу распознать chat_id %r, пропускаю.", part)
    return ids


async def _daily_news_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_ids: list[int] = context.application.bot_data.get("news_chat_ids", [])
    if not chat_ids:
        return

    service: OpenAIAssistantService = context.application.bot_data["assistant_service"]
    db: ThreadDatabase = context.application.bot_data["thread_db"]
    prompt: str = context.application.bot_data["news_prompt"]

    for chat_id in chat_ids:
        try:
            summary = await service.ask(chat_id=chat_id, user_text=prompt, db=db)
        except Exception:
            logger.exception("Не удалось собрать обзор для чата %s", chat_id)
            continue

        try:
            await context.bot.send_message(chat_id=chat_id, text=summary)
        except Exception:
            logger.exception("Не удалось отправить обзор в чат %s", chat_id)


def _schedule_daily_news(application: Application) -> None:
    chat_ids: list[int] = application.bot_data.get("news_chat_ids", [])
    if not chat_ids:
        return

    tz_name: str = application.bot_data.get("news_timezone", "")
    tzinfo = None
    if tz_name:
        try:
            tzinfo = ZoneInfo(tz_name)
        except Exception:
            logger.warning("Не удалось загрузить таймзону %s, использую локальное время.", tz_name)
    news_time = dt_time(hour=9, minute=0, tzinfo=tzinfo)
    application.job_queue.run_daily(
        _daily_news_job,
        news_time,
        name="daily_news",
    )
    logger.info("Ежедневный новостной обзор запланирован на %s в %s.", chat_ids, tz_name or "local")


async def _typing_loop(context: ContextTypes.DEFAULT_TYPE, chat_id: int, task: asyncio.Task[str]) -> None:
    while not task.done():
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            logger.exception("Failed to send typing action for chat_id=%s", chat_id)
        await asyncio.sleep(4)


async def run_with_delayed_typing(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    coro: Awaitable[str],
) -> str:
    task = asyncio.create_task(coro)
    typing_task: asyncio.Task[None] | None = None

    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=20.0)
    except asyncio.TimeoutError:
        typing_task = asyncio.create_task(_typing_loop(context, chat_id, task))
        return await task
    finally:
        if typing_task:
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task


async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.text:
        return

    bot_id: int = context.application.bot_data["bot_id"]
    bot_username: str = context.application.bot_data.get("bot_username", "")
    if not should_respond(update, bot_username=bot_username, bot_id=bot_id):
        return

    prompt = strip_bot_mention(message.text, bot_username=bot_username)
    today = datetime.now().strftime("%d.%m.%Y")
    prompt = f"Сегодня {today}: {prompt}"
    if not prompt:
        await message.reply_text("напиши хоть что-то после упоминания, а не пустоту.")
        return

    service: OpenAIAssistantService = context.application.bot_data["assistant_service"]
    db: ThreadDatabase = context.application.bot_data["thread_db"]

    try:
        response_text = await run_with_delayed_typing(
            context=context,
            chat_id=message.chat_id,
            coro=service.ask(chat_id=message.chat_id, user_text=prompt, db=db),
        )
        await message.reply_text(response_text)
    except (APITimeoutError, AssistantRunTimeoutError):
        logger.exception("OpenAI timeout for chat_id=%s", message.chat_id)
        await message.reply_text("зависло и не отвечает, попробуй позже.")
    except AssistantRunFailedError:
        logger.exception("OpenAI run failed for chat_id=%s", message.chat_id)
        await message.reply_text("не дожал твой запрос, кинь его еще раз.")
    except Exception:
        logger.exception("Unexpected handler error for chat_id=%s", message.chat_id)
        await message.reply_text("что-то сломалось, повтори позже.")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.error:
        logger.error(
            "Unhandled Telegram error. update=%s",
            update,
            exc_info=(
                type(context.error),
                context.error,
                context.error.__traceback__,
            ),
        )
        return
    logger.error("Unhandled Telegram error. update=%s", update)


async def post_init(application: Application) -> None:
    bot_user = await application.bot.get_me()
    application.bot_data["bot_id"] = bot_user.id
    application.bot_data["bot_username"] = bot_user.username or ""

    service: OpenAIAssistantService = application.bot_data["assistant_service"]
    await service.configure_assistant()

    _schedule_daily_news(application)

    logger.info(
        "Bot initialized. bot_id=%s, username=@%s",
        bot_user.id,
        bot_user.username,
    )


def main() -> None:
    load_dotenv()
    load_dotenv(BASE_DIR / ".env")
    setup_logging()

    telegram_token = _require_env("TELEGRAM_BOT_TOKEN")
    openai_api_key = _require_env("OPENAI_API_KEY")
    assistant_id = _require_env("OPENAI_ASSISTANT_ID")
    vector_store_id = os.getenv("OPENAI_VECTOR_STORE_ID")
    news_chat_ids = _parse_news_chat_ids(os.getenv("NEWS_CHAT_IDS"))
    news_prompt = os.getenv(
        "NEWS_PROMPT",
        "Составь короткий утренний обзор новостей (политика, экономика, технологии) в 3–4 пунктах.",
    )
    news_timezone = os.getenv("NEWS_TIMEZONE", "Europe/Moscow")

    db = ThreadDatabase(DB_PATH)
    service = OpenAIAssistantService(
        api_key=openai_api_key,
        config=AssistantConfig(
            assistant_id=assistant_id,
            vector_store_id=vector_store_id,
            temperature=1.0,
            run_timeout_seconds=120.0,
            poll_interval_seconds=1.0,
        ),
    )

    app = Application.builder().token(telegram_token).post_init(post_init).build()
    app.bot_data["thread_db"] = db
    app.bot_data["assistant_service"] = service
    app.bot_data["news_chat_ids"] = news_chat_ids
    app.bot_data["news_prompt"] = news_prompt
    app.bot_data["news_timezone"] = news_timezone

    app.add_handler(
        MessageHandler(
            filters.ChatType.GROUPS & filters.TEXT & ~filters.COMMAND,
            handle_group_message,
        )
    )
    app.add_error_handler(on_error)

    logger.info("Starting polling...")
    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()

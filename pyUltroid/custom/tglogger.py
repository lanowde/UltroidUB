# Original Source: https://github.com/subinps/tglogging
# Modified to make compatible with Ultroid.
#
# first push: 24-09-2022, v0.7
# fixes: 23-01-2023, v0.8
# pyrogram support: 15-01-2024

import asyncio
from io import BytesIO
from logging import StreamHandler

from ._loop import loop, run_async_task
from ._extras import async_searcher


_TG_MSG_LIMIT = 4000
_MAX_LOG_LIMIT = 12000
_PAYLOAD = {}


class TGLogHandlerBotAPI(StreamHandler):
    __slots__ = (
        "chat",
        "__tgtoken",
        "log_db",
        "current_log_msg",
        "message_id",
        "is_active",
        "sent_as_file",
        "_floodwait",
    )

    def __init__(self, chat, token):
        self.chat = chat
        self.__tgtoken = f"https://api.telegram.org/bot{token}"
        self.log_db = []
        self.current_log_msg = ""
        self.message_id = None
        self.is_active = False
        self._floodwait = False
        self.sent_as_file = False
        _PAYLOAD.update(
            {
                "chat_id": chat,
                "disable_web_page_preview": True,
                "parse_mode": "HTML",
            }
        )
        StreamHandler.__init__(self)

    async def _tg_logger(self):
        try:
            await asyncio.sleep(3)
            while self.log_db:
                await self.handle_logs(self.log_db.copy())
                await asyncio.sleep(8)
        finally:
            self.is_active = False

    def emit(self, record):
        msg = self.format(record)
        self.log_db.append("\n\n" + msg)
        if not (self.is_active or self._floodwait):
            self.is_active = True
            run_async_task(self._tg_logger, id="logger_task")

    def _splitter(self, logs):
        _log = []
        current = self.current_log_msg
        for l in logs:
            edit_left = _TG_MSG_LIMIT - len(current)
            if edit_left > len(l):
                current += l
            else:
                _log.append(current)
                current = l
        _log.append(current)
        return _log

    async def conditionX(self, log_msg):
        lst = self._splitter(log_msg)
        first_msg = lst.pop(0)
        if first_msg != self.current_log_msg:
            await self.edit_message(first_msg)
            await asyncio.sleep(8)
        for i in lst:
            await self.send_message(i)
            await asyncio.sleep(8)

    async def handle_logs(self, db):
        log_msgs = "".join(db)
        edit_left = _TG_MSG_LIMIT - len(self.current_log_msg)
        if edit_left > len(log_msgs):
            await self.edit_message(self.current_log_msg + log_msgs)
        elif any(len(i) > _TG_MSG_LIMIT for i in db) or len(log_msgs) > _MAX_LOG_LIMIT:
            await self.send_file(log_msgs)
        else:
            await self.conditionX(db)
        for _ in range(len(db)):
            self.log_db.pop(0)

    async def send_request(self, url, **kwargs):
        return await async_searcher(
            url, post=True, re_json=True, ssl=False, timeout=120, **kwargs
        )

    async def handle_floodwait(self, sleep):
        await asyncio.sleep(sleep)
        self._floodwait = False

    @staticmethod
    def _filter_text(text):
        text = text.lstrip()
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        return f"<code>{text}</code>"

    async def send_message(self, message):
        payload = _PAYLOAD.copy()
        if self.message_id:
            payload["reply_to_message_id"] = self.message_id
        payload["text"] = TGLogHandlerBotAPI._filter_text(message)
        res = await self.send_request(self.__tgtoken + "/sendMessage", json=payload)
        if res.get("ok"):
            self.message_id = int(res["result"]["message_id"])
            self.current_log_msg = message
            self.sent_as_file = False
        else:
            await self.handle_error(res)

    async def edit_message(self, message):
        if not self.message_id or self.sent_as_file:
            return await self.send_message(message)
        payload = _PAYLOAD.copy()
        payload["message_id"] = self.message_id
        payload["text"] = TGLogHandlerBotAPI._filter_text(message)
        res = await self.send_request(self.__tgtoken + "/editMessageText", json=payload)
        self.current_log_msg = message
        if not res.get("ok"):
            await self.handle_error(res)

    async def send_file(self, logs):
        payload = _PAYLOAD.copy()
        payload["caption"] = "Too much logs, hence sending as File."
        payload.pop("disable_web_page_preview", None)
        if self.message_id:
            payload["reply_to_message_id"] = self.message_id
        file = BytesIO(logs.lstrip().encode())
        file.name = "tglogging.txt"
        res = await self.send_request(
            self.__tgtoken + "/sendDocument",
            params=payload,
            data={"document": file},
        )
        if res.get("ok"):
            self.message_id = int(res["result"]["message_id"])
            self.current_log_msg = ""
            self.sent_as_file = True
        else:
            await self.handle_error(res)

    async def handle_error(self, resp):
        error = resp.get("parameters", {})
        if not error:
            description = resp.get("description")
            if resp.get("error_code") in (401, 400) and description.startswith(
                ("Unauthorized", "Bad Request: message is not modified")
            ):
                # same message content
                return
            elif resp.get("error_code") == 400 and any(
                i in description
                for i in (
                    "MESSAGE_ID_INVALID",
                    "Bad Request: message to edit not found",
                    "Bad Request: message to reply not found",
                )
            ):
                # deleted message
                self.message_id = None
            print(f"Errors while updating TG logs: {resp}")
            return

        elif s := error.get("retry_after"):
            self._floodwait = True
            print(f"tglogger: floodwait of {s}s")
            run_async_task(self.handle_floodwait, s + s // 4, id="tglogger_floodwait")


class PyroTGLogHandler(StreamHandler):
    __slots__ = (
        "ReplyParameters",
        "app",
        "chat",
        "log_db",
        "current_log_msg",
        "message_id",
        "is_active",
        "sent_as_file",
        "_floodwait",
    )

    def __init__(self, chat, token):
        self.chat = chat
        self.log_db = []
        self.current_log_msg = ""
        self.message_id = None
        self.is_active = False
        self._floodwait = False
        self.sent_as_file = False
        _PAYLOAD.update({"chat_id": chat, "disable_notification": True})
        self.initializer(bot_token=token)
        StreamHandler.__init__(self)

    def initializer(self, bot_token):
        from pyUltroid.custom._loop import loop

        print("Starting TGLogger Pyrogram Client!")
        loop.run_until_complete(self.client_startup(bot_token))

    def _post_startup_tasks(self):
        from pyrogram.types import LinkPreviewOptions, ReplyParameters
        from pyrogram.enums import ParseMode

        self.ReplyParameters = ReplyParameters
        _PAYLOAD.update(
            {
                "parse_mode": ParseMode.HTML,
                "link_preview_options": LinkPreviewOptions(is_disabled=True),
            }
        )

    async def client_startup(self, bot_token):
        try:
            from pyrogram import Client
        except ImportError:
            print("Install 'pyrogram' to use PyroTGLogger :)")
            quit(0)

        __import__("os").makedirs("resources/auth", exist_ok=True)

        self.app = Client(
            name="pyro_tglogger",
            api_id=6,
            api_hash="eb06d4abfb49dc3eeb1aeb98ae0f581e",
            bot_token=bot_token,
            workdir="resources/auth/",
            sleep_threshold=300,
            workers=1,
            no_updates=True,
            # in_memory=True,
            max_concurrent_transmissions=1,
            max_message_cache_size=8,  # kurigram
            max_topic_cache_size=8,
        )
        await self.app.start()
        from pyUltroid import _shutdown_tasks

        _shutdown_tasks.append(self.app.stop())
        self._post_startup_tasks()

    async def _tg_logger(self):
        try:
            await asyncio.sleep(3)
            while self.log_db:
                await self.handle_logs(self.log_db.copy())
                await asyncio.sleep(8)
        finally:
            self.is_active = False

    def emit(self, record):
        msg = self.format(record)
        self.log_db.append("\n\n" + msg)
        if not (self.is_active or self._floodwait):
            self.is_active = True
            run_async_task(self._tg_logger, id="pyro_logger_task")

    def _splitter(self, logs):
        _log = []
        current = self.current_log_msg
        for l in logs:
            edit_left = _TG_MSG_LIMIT - len(current)
            if edit_left > len(l):
                current += l
            else:
                _log.append(current)
                current = l
        _log.append(current)
        return _log

    async def conditionX(self, log_msg):
        lst = self._splitter(log_msg)
        first_msg = lst.pop(0)
        if first_msg != self.current_log_msg:
            await self.edit_message(first_msg)
            await asyncio.sleep(8)
        for i in lst:
            await self.send_message(i)
            await asyncio.sleep(8)

    async def handle_logs(self, db):
        log_msgs = "".join(db)
        edit_left = _TG_MSG_LIMIT - len(self.current_log_msg)
        if edit_left > len(log_msgs):
            await self.edit_message(self.current_log_msg + log_msgs)
        elif any(len(i) > _TG_MSG_LIMIT for i in db) or len(log_msgs) > _MAX_LOG_LIMIT:
            await self.send_file(log_msgs)
        else:
            await self.conditionX(db)
        for _ in range(len(db)):
            self.log_db.pop(0)

    async def handle_floodwait(self, sleep):
        await asyncio.sleep(sleep)
        self._floodwait = False

    @staticmethod
    def _filter_text(text):
        text = text.lstrip()
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        return f"<code>{text}</code>"

    async def send_message(self, message):
        payload = _PAYLOAD.copy()

        payload.update(
            {
                "reply_parameters": self.ReplyParameters(
                    message_id=getattr(self, "message_id", None)
                ),
                "text": PyroTGLogHandler._filter_text(message),
            }
        )
        try:
            msg = await self.app.send_message(**payload)
        except Exception as exc:
            await self.handle_error(exc)
        else:
            self.message_id = msg.id
            self.current_log_msg = message
            self.sent_as_file = False

    async def edit_message(self, message):
        if not self.message_id or self.sent_as_file:
            return await self.send_message(message)
        payload = _PAYLOAD.copy()
        payload["message_id"] = getattr(self, "message_id", None)
        payload["text"] = PyroTGLogHandler._filter_text(message)
        payload.pop("disable_notification", None)
        try:
            msg = await self.app.edit_message_text(**payload)
        except Exception as exc:
            await self.handle_error(exc)
        else:
            self.current_log_msg = message

    async def send_file(self, logs):
        payload = _PAYLOAD.copy()
        payload["caption"] = "Too much logs, hence sending as File."
        if msg_id := getattr(self, "message_id", None):
            payload["reply_parameters"] = self.ReplyParameters(message_id=msg_id)

        payload.pop("link_preview_options", None)
        file = BytesIO(logs.lstrip().encode())
        file.name = "tglogging.txt"
        payload["document"] = file
        try:
            msg = await self.app.send_document(**payload)
        except Exception as exc:
            await self.handle_error(exc)
        else:
            self.message_id = msg.id
            self.current_log_msg = ""
            self.sent_as_file = True

    async def handle_error(self, exc):
        from pyrogram import errors

        if isinstance(exc, errors.FloodWait):
            # floodwait
            self._floodwait = True
            s = exc.value
            print(f"tglogger: floodwait of {s}s")
            run_async_task(
                self.handle_floodwait, s + s // 4, id="pyro_tglogger_floodwait"
            )
        elif isinstance(exc, errors.MessageIdInvalid):
            # deleted message
            self.message_id = None
        elif isinstance(exc, errors.MessageNotModified):
            # same message content
            return
        else:
            print(f"Errors while updating TG logs: {exc}")

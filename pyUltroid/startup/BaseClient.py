# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

__all__ = ("UltroidClient",)

import inspect
import sys
import time
from logging import Logger
from pathlib import Path

from telethon import TelegramClient
from telethon import utils as telethon_utils
from telethon.errors import (
    AccessTokenExpiredError,
    AccessTokenInvalidError,
    ApiIdInvalidError,
    AuthKeyDuplicatedError,
    MessageNotModifiedError,
    MessageIdInvalidError,
)

from ._logger import TelethonLogger
from . import *
from pyUltroid.configs import Var
from pyUltroid.exceptions import DownloadError, UploadError
from pyUltroid.custom.FastTelethon import download_file, upload_file
from pyUltroid.custom.commons import progress, check_filename, get_tg_filename


class UltroidClient(TelegramClient):
    def __init__(
        self,
        session,
        api_id=None,
        api_hash=None,
        bot_token=None,
        udB=None,
        logger: Logger = LOGS,
        log_attempt=True,
        exit_on_error=True,
        *args,
        **kwargs,
    ):
        self._cache = {}
        self._dialogs = []
        self._handle_error = exit_on_error
        self._log_at = log_attempt
        self.logger = logger
        self.udB = udB
        kwargs["api_id"] = api_id or Var.API_ID
        kwargs["api_hash"] = api_hash or Var.API_HASH
        kwargs["base_logger"] = TelethonLogger
        super().__init__(session, **kwargs)
        self.run_in_loop(self.start_client(bot_token=bot_token))
        self.dc_id = self.session.dc_id

    def __repr__(self):
        return f"<Ultroid.Client :\n self: {self.full_name}\n bot: {self.me.bot}\n>"

    @property
    def __dict__(self):
        if self.me:
            return self.me.to_dict()

    async def start_client(self, **kwargs):
        """function to start client"""
        if self._log_at:
            self.logger.info("Trying to login.")

        try:
            await self.start(**kwargs)
        except ApiIdInvalidError:
            self.logger.critical("API ID and API_HASH combination does not match!")

            sys.exit()
        except (AuthKeyDuplicatedError, EOFError) as er:
            if self._handle_error:
                self.logger.critical("String session expired. Create new!")
                return sys.exit()
            self.logger.critical("String session expired.")
        except (AccessTokenExpiredError, AccessTokenInvalidError):
            # AccessTokenError can only occur for Bot account
            # And at Early Process, Its saved in DB.
            self.udB.del_key("BOT_TOKEN")
            self.logger.critical(
                "Bot token is expired or invalid. Create new from @Botfather and add in BOT_TOKEN env variable!"
            )
            sys.exit()

        # Save some stuff for later use...
        self.me = await self.get_me()
        if self.me.bot:
            me = f"@{self.me.username}"
        else:
            setattr(self.me, "phone", None)
            me = self.full_name
        if self._log_at:
            self.logger.info(f"Logged in as {me}")
        self._bot = await self.is_bot()

    async def fast_uploader(self, file, **kwargs):
        """Upload files in a faster way"""
        path = Path(file)
        filename = kwargs.get("filename", path.name)
        # Set to True and pass event to show progress bar.
        show_progress = kwargs.get("show_progress", False)
        if show_progress:
            event = kwargs["event"]
        # Whether to use cached file for uploading or not
        use_cache = kwargs.get("use_cache", True)
        # Delete original file after uploading
        to_delete = kwargs.get("to_delete", False)
        message = kwargs.get("message", f"Uploading {filename}...")
        by_bot = self._bot
        size = path.stat().st_size
        start_time = time.time()
        # Don't show progress bar when file size is less than 5MB.
        if size < 5 * 2**20:
            show_progress = False
        if use_cache and self._cache and (_cache := self._cache.get("upload_cache")):
            for files in _cache:
                if (
                    files["size"] == size
                    and files["path"] == path
                    and files["name"] == filename
                    and files["by_bot"] == by_bot
                ):
                    if to_delete:
                        path.unlink(missing_ok=True)
                    return files["raw_file"], time.time() - start_time

        raw_file, edit_missed = None, 0
        while not raw_file:
            with open(file, "rb") as f:
                try:
                    raw_file = await upload_file(
                        client=self,
                        file=f,
                        filename=filename,
                        progress_callback=(
                            lambda completed, total: self.loop.create_task(
                                progress(completed, total, event, start_time, message)
                            )
                        )
                        if show_progress
                        else None,
                    )
                except MessageNotModifiedError as exc:
                    edit_missed += 1
                    if edit_missed >= 6:
                        raise UploadError(exc) from None
                except MessageIdInvalidError:
                    raise UploadError(
                        f"Upload Cancelled for '{file}' because message was deleted."
                    ) from None

        if kwargs.get("save_cache", True):
            cache = {
                "by_bot": by_bot,
                "size": size,
                "path": path,
                "name": filename,
                "raw_file": raw_file,
            }
            _cache = self._cache.get("upload_cache")
            self._cache["upload_cache"] = _cache.append(cache) if _cache else [cache]
        if to_delete:
            path.unlink(missing_ok=True)

        return raw_file, time.time() - start_time

    async def fast_downloader(self, file, **kwargs):
        """Download files in a faster way"""
        # Set to True and pass event to show progress bar.
        show_progress = kwargs.get("show_progress", False)
        filename = kwargs.get("filename", None)
        if show_progress:
            event = kwargs["event"]
        # Don't show progress bar when file size is less than 10MB.
        if file.size < 10 * 2**20:
            show_progress = False

        # Auto-generate Filename
        filename = check_filename(filename or get_tg_filename(file))
        message = kwargs.get("message", f"Downloading {filename}...")
        raw_file, edit_missed = None, 0
        start_time = time.time()

        while not raw_file:
            with open(filename, "wb") as f:
                try:
                    raw_file = await download_file(
                        client=self,
                        location=file,
                        out=f,
                        progress_callback=(
                            lambda completed, total: self.loop.create_task(
                                progress(completed, total, event, start_time, message)
                            )
                        )
                        if show_progress
                        else None,
                    )
                except MessageNotModifiedError as exc:
                    edit_missed += 1
                    if edit_missed >= 6:
                        raise DownloadError(exc) from None
                except MessageIdInvalidError:
                    raise DownloadError(
                        f"Download Cancelled for '{filename}' because message was deleted."
                    ) from None

        return raw_file, time.time() - start_time

    def run_in_loop(self, function):
        """run inside asyncio loop"""
        return self.loop.run_until_complete(function)

    def run(self):
        """run asyncio loop"""
        self.run_until_disconnected()

    def add_handler(self, func, *args, **kwargs):
        """Add new event handler, ignoring if exists"""
        for i in self.list_event_handlers():
            if func == i[0]:
                return
        self.add_event_handler(func, *args, **kwargs)

    @property
    def utils(self):
        return telethon_utils

    @property
    def full_name(self):
        """full name of Client"""
        return self.utils.get_display_name(self.me)

    @property
    def uid(self):
        """Client's user id"""
        return self.me.id

    def to_dict(self):
        return dict(inspect.getmembers(self))

    async def parse_id(self, text):
        try:
            text = int(text)
        except ValueError:
            pass
        return await self.get_peer_id(text)

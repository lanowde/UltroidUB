# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

import inspect
import re
from traceback import format_exc

from telethon import Button
from telethon.errors import QueryIdInvalidError, ResultIdDuplicateError
from telethon.events import CallbackQuery, InlineQuery, NewMessage
from telethon.tl.types import InputWebDocument

from .. import LOGS, asst, udB, ultroid_bot
from ..fns.admins import admin_check
from . import SUDO_M, append_or_update, owner_and_sudos


OWNER = ultroid_bot.full_name

MSG = f"""
**Ultroid - UserBot**
➖➖➖➖➖➖➖➖➖➖
**Owner**: [{OWNER}](tg://user?id={ultroid_bot.uid})
**Support**: @TeamUltroid
➖➖➖➖➖➖➖➖➖➖
"""

IN_BTTS = [
    [
        Button.url(
            "Repository",
            url="https://github.com/TeamUltroid/Ultroid",
        ),
        Button.url("Support", url="https://t.me/UltroidSupportChat"),
    ]
]


# decorator for assistant
def asst_cmd(pattern=None, load=None, owner=False, **kwargs):
    """Decorator for assistant's command"""
    name = inspect.stack()[1].filename.split("/")[-1].replace(".py", "")
    kwargs["forwards"] = False

    def ult(func):
        if pattern:
            kwargs["pattern"] = re.compile(f"^/{pattern}")

        async def handler(event):
            if owner and event.sender_id not in owner_and_sudos():
                return

            try:
                await func(event)
            except Exception as er:
                LOGS.exception(er)

        asst.add_event_handler(handler, NewMessage(**kwargs))
        if load is not None:
            append_or_update(load, func, name, kwargs)

    return ult


# callback decorator for assistant
def callback(data=None, from_users=[], admins=False, owner=False, **kwargs):
    """Assistant's callback decorator"""
    full_sudo = kwargs.pop("fullsudo", None)
    if "me" in from_users:
        from_users.remove("me")
        from_users.append(ultroid_bot.uid)

    def ultr(func):
        async def wrapper(event):
            if admins and not await admin_check(event):
                return await event.answer()
            if from_users and event.sender_id not in from_users:
                return await event.answer("Not for You!", alert=True)
            if (full_sudo and event.sender_id not in SUDO_M.fullsudos) or (
                owner and event.sender_id not in owner_and_sudos()
            ):
                return await event.answer(f"This is {OWNER}'s bot!!")

            try:
                await func(event)
            except Exception as er:
                LOGS.exception(er)

        asst.add_event_handler(wrapper, CallbackQuery(data=data, **kwargs))

    return ultr


# inline decorator for assistant
def in_pattern(pattern=None, owner=False, **kwargs):
    """Assistant's inline decorator."""
    full_sudo = kwargs.pop("fullsudo", None)

    def don(func):
        async def wrapper(event):
            if (full_sudo and event.sender_id not in SUDO_M.fullsudos) or (
                owner and event.sender_id not in owner_and_sudos()
            ):
                img = "https://graph.org/file/dde85d441fa051a0d7d1d.jpg"
                res = [
                    await event.builder.article(
                        title="Ultroid Userbot",
                        url="https://t.me/TeamUltroid",
                        description="(c) TeamUltroid",
                        text=MSG,
                        thumb=InputWebDocument(img, 0, "image/jpeg", []),
                        buttons=IN_BTTS,
                    )
                ]
                return await event.answer(
                    res,
                    switch_pm=f"🤖: Assistant of {OWNER}",
                    switch_pm_param="start",
                )

            try:
                await func(event)
            except (QueryIdInvalidError, ResultIdDuplicateError):
                pass
            except Exception as er:
                err = format_exc()

                def error_text():
                    return f"**#ERROR #INLINE**\n\nQuery: `{asst.me.username} {pattern}`\n\n**Traceback:**\n`{format_exc()}`"

                LOGS.exception(er)
                try:
                    await event.answer(
                        [
                            await event.builder.article(
                                title="Unhandled Exception has Occured!",
                                text=error_text(),
                                buttons=Button.url(
                                    "Report", "https://t.me/UltroidSupportChat"
                                ),
                            )
                        ]
                    )
                except (QueryIdInvalidError, ResultIdDuplicateError):
                    LOGS.exception(err)
                except Exception as er:
                    LOGS.exception(er)
                    await asst.send_message(udB.get_key("LOG_CHANNEL"), error_text())

        asst.add_event_handler(wrapper, InlineQuery(pattern=pattern, **kwargs))

    return don

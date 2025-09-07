# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

__all__ = ("asst_cmd", "callback", "in_pattern")

import inspect
import re
from pathlib import Path
from traceback import format_exc

from telethon import Button
from telethon.errors import QueryIdInvalidError, ResultIdDuplicateError
from telethon.events import CallbackQuery, InlineQuery, NewMessage
from telethon.tl.types import InputWebDocument, PeerChat, PeerChannel, PeerUser

from pyUltroid import LOGS, asst, udB, ultroid_bot
from pyUltroid.dB._core import LOADED
from pyUltroid.custom.commons import not_so_fast
from pyUltroid.fns.admins import admin_check
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

    def asstcmd_wrap(func):
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

    return asstcmd_wrap


# logger for callback query events
async def _callback_logger(event, out_chat):
    try:
        if event.via_inline:
            equery = event.query
            btns = None
            dc = getattr(equery.msg_id, "dc_id", 0)
            fmt_msg = f"<b>#callback #via_inline clicked by</b> <code>{equery.user_id}</code>\n\n<b>Query:</b>  <code>{equery.data.decode()}</code> \n<b>Chat DC: {dc}</b>\n"
            if owner_id := getattr(equery.msg_id, "owner_id", None):
                # callback via inline in pvt chat.
                if equery.user_id != owner_id:
                    # message owned by someone; while clicked by someone else
                    fmt_msg += f" \n<b>Message Owner ID:</b>  <code>{owner_id}</code>"
                # only applicable for the user
                fmt_msg += (
                    f" \n<b>Pvt. Message ID:</b>  <code>{equery.msg_id.id}</code>"
                )
        else:
            equery = event.query
            if isinstance(equery.peer, PeerChannel):
                chat_id = getattr(equery.peer, "channel_id", 0)
            elif isinstance(equery.peer, PeerUser):
                chat_id = getattr(equery.peer, "user_id", 0)
            elif isinstance(equery.peer, PeerChat):
                chat_id = getattr(equery.peer, "chat_id", 0)
            else:
                chat_id = 0  # should not happen

            fmt_msg = f"<b>#callback event clicked by</b> <code>{equery.user_id}</code>\n\n>>  {equery.data.decode()}"
            btns = [
                Button.url(
                    "message link!  *unreliable*",
                    url=f"https://t.me/c/{chat_id}/{equery.msg_id}",
                )
            ]

        await not_so_fast(
            asst.send_message,
            out_chat,
            fmt_msg,
            buttons=btns,
            sleep=6.5,
            parse_mode="html",
            link_preview=False,
        )
    except Exception as exc:
        LOGS.exception(f"Error while logging callback event: {exc}")


# callback decorator for assistant
def callback(data=None, from_users=[], admins=False, owner=False, **kwargs):
    """Assistant's callback decorator"""
    full_sudo = kwargs.pop("fullsudo", None)
    if "me" in from_users:
        from_users.remove("me")
        from_users.append(ultroid_bot.uid)

    def callback_wrap(func):
        async def callback_wrapper(event):
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
            finally:
                if out_chat := udB.get_key("LOG_CALLBACK_COMMANDS"):
                    await _callback_logger(event, out_chat)

        asst.add_event_handler(callback_wrapper, CallbackQuery(data=data, **kwargs))

        _file = inspect.stack()[1].filename
        if "addons/" in _file:
            stem = Path(_file).stem
            try:
                LOADED[stem].append(callback_wrapper)
            except KeyError:
                LOADED[stem] = [callback_wrapper]

    return callback_wrap


# logger for inline events
async def _inline_logger(event, out_chat):
    try:
        page = int(event.query.offset or 0) + 1
        fmt_msg = f"#inline triggered by <code>{event.sender_id}</code>; [Page: {page}]\n>>  {event.text[:4000]}"
        await not_so_fast(
            asst.send_message,
            out_chat,
            fmt_msg,
            sleep=6.5,
            parse_mode="html",
            link_preview=False,
        )
    except Exception as exc:
        LOGS.exception(f"Error while logging inline commands: {exc}")


# inline decorator for assistant
def in_pattern(pattern=None, owner=False, **kwargs):
    """Assistant's inline decorator."""
    full_sudo = kwargs.pop("fullsudo", None)

    def inline_wrap(func):
        async def inline_wrapper(event):
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

                error_text = (
                    lambda: f"**#ERROR #INLINE**\n\nQuery: `{asst.me.username} {event.text}`\n\n**Traceback:**\n`{format_exc()}`"
                )

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
                    pass
                except Exception as er:
                    LOGS.exception(er)
                    await asst.send_message(udB.get_key("LOG_CHANNEL"), error_text())
            finally:
                if out_chat := udB.get_key("LOG_INLINE_COMMANDS"):
                    await _inline_logger(event, out_chat)

        asst.add_event_handler(inline_wrapper, InlineQuery(pattern=pattern, **kwargs))

        _file = inspect.stack()[1].filename
        if "addons/" in _file:
            stem = Path(_file).stem
            try:
                LOADED[stem].append(inline_wrapper)
            except KeyError:
                LOADED[stem] = [inline_wrapper]

    return inline_wrap

# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

__all__ = ("ultroid_cmd",)

import asyncio
import inspect
import re
import sys
from io import BytesIO
from functools import wraps
from pathlib import Path
from time import gmtime, sleep, strftime
from traceback import format_exc

from telethon import Button
from telethon import __version__ as telever
from telethon import events
from telethon.errors.common import AlreadyInConversationError
from telethon.errors.rpcerrorlist import (
    AuthKeyDuplicatedError,
    BotInlineDisabledError,
    BotMethodInvalidError,
    ChatSendInlineForbiddenError,
    ChatSendMediaForbiddenError,
    ChatSendStickersForbiddenError,
    FloodWaitError,
    MessageDeleteForbiddenError,
    MessageIdInvalidError,
    MessageNotModifiedError,
    UserIsBotError,
)
from telethon.events import MessageEdited, NewMessage
from telethon.tl.types import MessageMediaWebPage
from telethon.utils import get_display_name

from pyUltroid import *
from pyUltroid import _ignore_eval
from strings import get_string
from pyUltroid.dB import DEVLIST
from pyUltroid.dB._core import LIST, LOADED
from pyUltroid.fns.admins import admin_check
from pyUltroid.custom.commons import bash, not_so_fast, time_formatter as tf
from pyUltroid.exceptions import DependencyMissingError
from pyUltroid.version import __version__ as pyver, ultroid_version as ult_ver
from . import SUDO_M, owner_and_sudos
from ._wrappers import eod


MANAGER = udB.get_key("MANAGER")
TAKE_EDITS = udB.get_key("TAKE_EDITS")
TAKE_SUDO_EDITS = udB.get_key("TAKE_SUDO_EDITS")
TAKE_ASST_EDITS = udB.get_key("TAKE_ASST_EDITS")
black_list_chats = udB.get_key("BLACKLIST_CHATS")
allow_sudo = SUDO_M.should_allow_sudo


def compile_pattern(data, hndlr):
    if data.startswith("^"):
        data = data[1:]
    if data.startswith("."):
        data = data[1:]
    if hndlr in (" ", "NO_HNDLR"):
        # No Hndlr Feature
        return re.compile("^" + data)
    return re.compile("\\" + hndlr + data)


def _add_func_to_loaded(func, file):
    if "addons/" in str(file):
        try:
            LOADED[file.stem].append(func)
        except KeyError:
            LOADED[file.stem] = [func]


async def _log_sudo_commands(ult, chat, out_chat):
    try:
        await asyncio.sleep(3)
        sender = get_display_name(ult.sender or await ult.get_sender())
        fmt_msg = f"<b>#bot Command Executed by {sender}</b> [<code>{ult.sender_id}</code>]\n\n>> {ult.text[:4000]}"
        btns = [Button.url(get_display_name(chat), url=ult.message_link)]
        await not_so_fast(
            asst.send_message,
            out_chat,
            fmt_msg, buttons=btns, sleep=5, parse_mode="html", link_preview=False,
        )
    except Exception as exc:
        LOGS.exception(f"Error while logging sudo commands: {exc}")


def ultroid_cmd(
    pattern=None, manager=False, ultroid_bot=ultroid_bot, asst=asst, **kwargs
):
    owner_only = kwargs.get("owner_only", False)
    groups_only = kwargs.get("groups_only", False)
    admins_only = kwargs.get("admins_only", False)
    fullsudo = kwargs.get("fullsudo", False)
    only_devs = kwargs.get("only_devs", False)

    incoming_func = lambda e: not (
        (e.media and not isinstance(e.media, MessageMediaWebPage)) or e.via_bot_id
    )

    func = kwargs.get("func", incoming_func)

    def out_wrapper(dec):
        @wraps(dec)
        async def in_wrapper(ult):
            if not ult.out:
                if ult.sender_id not in owner_and_sudos():
                    return
                elif fullsudo and ult.sender_id not in SUDO_M.fullsudos:
                    return await eod(ult, get_string("py_d2"), time=15)
                elif ult.sender_id in _ignore_eval:
                    return await eod(
                        ult,
                        get_string("py_d1"),
                    )
                elif owner_only:
                    return

            if only_devs and not udB.get_key("I_DEV"):
                return await eod(
                    ult,
                    get_string("py_d4").format(HNDLR),
                    time=10,
                )

            chat = ult.chat

            """
            # not necessary; just makes bot slow!
            if not ult.is_private and hasattr(chat, "title"):
                if (
                    "#noub" in chat.title.lower()
                    and not (chat.admin_rights or chat.creator)
                    and not (ult.sender_id in DEVLIST)
                ):
                    return
            """

            if ult.is_private and (groups_only or admins_only):
                return await eod(ult, get_string("py_d3"))
            elif admins_only and not (chat.admin_rights or chat.creator):
                return await eod(ult, get_string("py_d5"))

            try:
                await dec(ult)
            except FloodWaitError as fwerr:
                client = ultroid_bot if ult.client._bot else asst
                await client.send_message(
                    udB.get_key("LOG_CHANNEL"),
                    f"`FloodWaitError:\n{str(fwerr)}\n\nBot Sleeping for {tf((fwerr.seconds + 15) * 1000)}`",
                )
                # await ultroid_bot.disconnect()
                sleep(fwerr.seconds + 20)
                # await ultroid_bot.connect()
                await client.send_message(
                    udB.get_key("LOG_CHANNEL"),
                    "`Bot is working again!!`",
                )
                return
            except ChatSendInlineForbiddenError:
                return await eod(ult, "`Inline Locked In This Chat.`")
            except (ChatSendMediaForbiddenError, ChatSendStickersForbiddenError):
                return await eod(ult, get_string("py_d8"))
            except (BotMethodInvalidError, UserIsBotError):
                return await eod(ult, get_string("py_d6"))
            except AlreadyInConversationError:
                return await eod(ult, get_string("py_d7"))
            except (BotInlineDisabledError, DependencyMissingError) as er:
                return await eod(ult, f"`{er}`")
            except (
                MessageIdInvalidError,
                MessageNotModifiedError,
                MessageDeleteForbiddenError,
            ) as er:
                LOGS.exception(er)
            except AuthKeyDuplicatedError as er:
                LOGS.exception(er)
                await asst.send_message(
                    udB.get_key("LOG_CHANNEL"),
                    "Session String expired, create new session from 👇",
                    buttons=[
                        Button.url("Bot", "t.me/SessionGeneratorBot?start="),
                        Button.url(
                            "Repl",
                            "https://replit.com/@TheUltroid/UltroidStringSession",
                        ),
                    ],
                )
                sys.exit()
            except events.StopPropagation:
                raise events.StopPropagation
            except KeyboardInterrupt:
                pass
            except Exception as e:
                LOGS.exception(e)
                date = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                naam = get_display_name(chat)
                ftext = "**Ultroid Client Error:** `Forward this to` @UltroidSupportChat\n\n"
                ftext += "**Py-Ultroid Version:** `" + str(pyver)
                ftext += "`\n**Ultroid Version:** `" + str(ult_ver)
                ftext += "`\n**Telethon Version:** `" + str(telever)
                ftext += f"`\n**Hosted At:** `{HOSTED_ON}`\n\n"
                ftext += "--------START ULTROID CRASH LOG--------"
                ftext += "\n**Date:** `" + date
                ftext += "`\n**Group:** `" + str(ult.chat_id) + "` " + str(naam)
                ftext += "\n**Sender ID:** `" + str(ult.sender_id)
                ftext += "`\n**Replied:** `" + str(ult.is_reply)
                ftext += "`\n\n**Event Trigger:**`\n"
                ftext += str(ult.text)
                ftext += "`\n\n**Traceback info:**`\n"
                ftext += str(format_exc())
                ftext += "`\n\n**Error text:**`\n"
                ftext += str(sys.exc_info()[1])
                ftext += "`\n\n--------END ULTROID CRASH LOG--------"
                ftext += "\n\n\n**Last 5 commits:**`\n"

                stdout, stderr = await bash('git log --pretty=format:"%an: %s" -5')
                result = stdout + (stderr or "")

                ftext += f"{result}`"

                if len(ftext) > 4096:
                    with BytesIO(ftext.encode()) as file:
                        file.name = "logs.txt"
                        error_log = await asst.send_file(
                            udB.get_key("LOG_CHANNEL"),
                            file,
                            caption="**Ultroid Client Error:** `Forward this to` @UltroidSupportChat\n\n",
                        )
                else:
                    error_log = await asst.send_message(
                        udB.get_key("LOG_CHANNEL"),
                        ftext,
                    )
                if ult.out:
                    await ult.edit(
                        f"<b><a href={error_log.message_link}>[An error occurred]</a></b>",
                        link_preview=False,
                        parse_mode="html",
                    )
            finally:
                if not ult.out and (out_chat := udB.get_key("LOG_SUDO_COMMANDS")):
                    await _log_sudo_commands(ult, chat, out_chat)

        cmd = None
        blacklist_chats = False
        chats = None
        if black_list_chats:
            blacklist_chats = True
            chats = list(black_list_chats)

        # incoming handler for sudo users..
        if allow_sudo:
            if pattern:
                cmd = compile_pattern(pattern, SUDO_HNDLR)
            ultroid_bot.add_event_handler(
                in_wrapper,
                NewMessage(
                    incoming=True,
                    pattern=cmd,
                    forwards=False,
                    func=func,
                    chats=chats,
                    blacklist_chats=blacklist_chats,
                ),
            )

        # incoming handler for owner
        if pattern:
            cmd = compile_pattern(pattern, HNDLR)
        ultroid_bot.add_event_handler(
            in_wrapper,
            NewMessage(
                outgoing=True,
                pattern=cmd,
                forwards=False,
                func=func,
                chats=chats,
                blacklist_chats=blacklist_chats,
            ),
        )

        # is_channel refers to megagroup ;_;
        # ignore edits that are -
        #  - triggered by reaction,
        #  - or message is older than 30 minutes.
        edit_func = lambda e: incoming_func(e) and not (
            (e.is_channel and e.chat.broadcast)
            or getattr(e.message, "edit_hide", None)
            or (e.message.edit_date - e.message.date).seconds > 1800
        )

        # edited message handler for sudo users
        # don't add handler if there is no pattern.
        if allow_sudo and TAKE_SUDO_EDITS and pattern:
            cmd = compile_pattern(pattern, SUDO_HNDLR)
            ultroid_bot.add_event_handler(
                in_wrapper,
                MessageEdited(
                    incoming=True,
                    pattern=cmd,
                    forwards=False,
                    func=edit_func,
                    chats=chats,
                    blacklist_chats=blacklist_chats,
                ),
            )

        # edited message handler for owner
        # don't add handler if there is no pattern.
        if TAKE_EDITS and pattern:
            cmd = compile_pattern(pattern, HNDLR)
            ultroid_bot.add_event_handler(
                in_wrapper,
                MessageEdited(
                    outgoing=True,
                    pattern=cmd,
                    forwards=False,
                    func=edit_func,
                    chats=chats,
                    blacklist_chats=blacklist_chats,
                ),
            )

        # manager plugin
        if manager and MANAGER:
            allow_all = kwargs.get("allow_all", False)
            allow_pm = kwargs.get("allow_pm", False)
            require = kwargs.get("require", None)

            async def manager_cmd(ult):
                if not allow_all and not (await admin_check(ult, require=require)):
                    return
                if not allow_pm and ult.is_private:
                    return

                try:
                    await dec(ult)
                except Exception as er:
                    if chat := udB.get_key("MANAGER_LOG"):
                        text = f"**#MANAGER_LOG\n\nChat:** `{get_display_name(ult.chat)}` `{ult.chat_id}`"
                        text += f"\n**Replied :** `{ult.is_reply}`\n**Command :** {ult.text}\n\n**Error :** `{er}`"
                        try:
                            return await asst.send_message(
                                chat, text, link_preview=False
                            )
                        except Exception as er:
                            LOGS.exception(er)
                    LOGS.info(f"• MANAGER [{ult.chat_id}]:")
                    LOGS.exception(er)

            if pattern:
                cmd = compile_pattern(pattern, "/")
            asst.add_event_handler(
                manager_cmd,
                NewMessage(
                    incoming=True,
                    pattern=cmd,
                    forwards=False,
                    func=func,
                    chats=chats,
                    blacklist_chats=blacklist_chats,
                ),
            )

        if DUAL_MODE and not (manager and DUAL_HNDLR == "/"):
            if pattern:
                cmd = compile_pattern(pattern, DUAL_HNDLR)
            asst.add_event_handler(
                in_wrapper,
                NewMessage(
                    incoming=True,
                    pattern=cmd,
                    forwards=False,
                    func=func,
                    chats=chats,
                    blacklist_chats=blacklist_chats,
                ),
            )
            if TAKE_ASST_EDITS and pattern:
                asst.add_event_handler(
                    in_wrapper,
                    MessageEdited(
                        incoming=True,
                        pattern=cmd,
                        forwards=False,
                        func=edit_func,
                        chats=chats,
                        blacklist_chats=blacklist_chats,
                    ),
                )

        file = Path(inspect.stack()[1].filename)
        _add_func_to_loaded(in_wrapper, file)
        if manager and MANAGER:
            _add_func_to_loaded(manager_cmd, file)
        if pattern:
            try:
                LIST[file.stem].append(pattern)
            except KeyError:
                LIST[file.stem] = [pattern]

        return in_wrapper

    return out_wrapper

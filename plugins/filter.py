# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

from . import get_help

__doc__ = get_help("help_filter")

import os
import re

from telethon.tl.types import User
from telethon.utils import pack_bot_file_id

from pyUltroid.dB.filter_db import add_filter, get_filter, list_filter, rem_filter
from pyUltroid.fns.tools import create_tl_btn, format_btn, get_msg_button

from . import events, get_string, mediainfo, not_so_fast, udB, ultroid_bot, ultroid_cmd
from ._inline import something


@ultroid_cmd(pattern="addfilter( (.*)|$)")
async def af(e):
    wrd = (e.pattern_match.group(1).strip()).lower()
    wt = await e.get_reply_message()
    chat = e.chat_id
    if not (wt and wrd):
        return await e.eor(get_string("flr_1"))
    btn = format_btn(wt.buttons) if wt.buttons else None
    if wt and wt.media:
        wut = mediainfo(wt.media)
        if wut.startswith(("pic", "gif")):
            dl = await wt.download_media()
            variable = await TelegraphClient.upload_file(dl, anon=True)
            m = f"https://graph.org{variable[0]}"
        elif wut == "video":
            if wt.media.document.size > 6 * 1000 * 1000:
                return await e.eor(get_string("com_4"), time=5)
            dl = await wt.download_media()
            variable = await TelegraphClient.upload_file(dl, anon=True)
            os.remove(dl)
            m = f"https://graph.org{variable[0]}"
        else:
            m = pack_bot_file_id(wt.media)
        if wt.text:
            txt = wt.text
            if not btn:
                txt, btn = get_msg_button(wt.text)
            add_filter(chat, wrd, txt, m, btn)
        else:
            add_filter(chat, wrd, None, m, btn)
    else:
        txt = wt.text
        if not btn:
            txt, btn = get_msg_button(wt.text)
        add_filter(chat, wrd, txt, None, btn)
    await e.eor(get_string("flr_4").format(wrd))
    ultroid_bot.add_handler(
        filter_func, events.NewMessage(func=lambda e: e.text and not e.media)
    )


@ultroid_cmd(pattern="remfilter( (.*)|$)")
async def rf(e):
    wrd = (e.pattern_match.group(1).strip()).lower()
    chat = e.chat_id
    if not wrd:
        return await e.eor(get_string("flr_3"))
    rem_filter(int(chat), wrd)
    await e.eor(get_string("flr_5").format(wrd))


@ultroid_cmd(pattern="listfilter$")
async def lsnote(e):
    if x := list_filter(e.chat_id):
        sd = "Filters Found In This Chats Are\n\n"
        return await e.eor(sd + x)
    await e.eor(get_string("flr_6"))


async def filter_func(e):
    if e.sender and getattr(e.sender, "bot", None):
        return
    xx = (e.text).lower()
    chat = e.chat_id
    if x := get_filter(chat):
        for c in x:
            pat = r"( |^|[^\w])" + re.escape(c) + r"( |$|[^\w])"
            if re.search(pat, xx):
                if k := x.get(c):
                    msg = k["msg"]
                    media = k["media"]
                    if k.get("button"):
                        btn = create_tl_btn(k["button"])
                        return await something(e, msg, media, btn)
                    await not_so_fast(e.reply, msg, file=media, sleep=5)


if udB.get_key("FILTERS"):
    ultroid_bot.add_handler(
        filter_func, events.NewMessage(func=lambda e: e.text and not e.media)
    )

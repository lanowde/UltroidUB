# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

"""
✘ Commands Available -

• `{i}addnote <word><reply to a message>`
    add note in the used chat with replied message and choosen word.

• `{i}remnote <word>`
    Remove the note from used chat.

• `{i}listnote`
    list all notes.

• Use :
   set notes in group so all can use it.
   type `#(Keyword of note)` to get it
"""

import os

from telethon import events
from telethon.utils import pack_bot_file_id

from pyUltroid.dB.notes_db import add_note, get_notes, list_note, rem_note
from pyUltroid.fns.tools import create_tl_btn, format_btn, get_msg_button

from . import (
    Catbox,
    get_string,
    mediainfo,
    not_so_fast,
    udB,
    ultroid_bot,
    ultroid_cmd,
)
from ._inline import something


async def notes(e):
    xx = [z.replace("#", "") for z in e.text.lower().split() if z.startswith("#")]
    for word in xx:
        if k := get_notes(e.chat_id, word):
            msg = k["msg"]
            media = k["media"]
            if k.get("button"):
                btn = create_tl_btn(k["button"])
                return await something(e, msg, media, btn)
            await not_so_fast(
                e.client.send_message,
                e.chat_id,
                msg,
                file=media,
                reply_to=e.reply_to_msg_id or e.id,
                sleep=5,
            )


def is_enabled():
    for func, _ in ultroid_bot.list_event_handlers():
        if func == notes:
            return True


@ultroid_cmd(pattern="addnote( (.*)|$)", admins_only=True)
async def ad_n(e):
    wrd = (e.pattern_match.group(1).strip()).lower()
    wt = await e.get_reply_message()
    chat = e.chat_id
    if not (wt and wrd):
        return await e.eor(get_string("notes_1"), time=5)
    if "#" in wrd:
        wrd = wrd.replace("#", "")
    btn = format_btn(wt.buttons) if wt.buttons else None
    if wt and wt.media:
        wut = mediainfo(wt.media)
        if wut.startswith(("pic", "gif")):
            dl = await wt.download_media()
            m = await Catbox(dl)
            os.remove(dl)
        elif wut == "video":
            if wt.media.document.size > 20 * 1000 * 1000:
                return await e.eor(get_string("com_4"), time=5)
            dl = await wt.download_media()
            m = await Catbox(dl)
            os.remove(dl)
        else:
            m = pack_bot_file_id(wt.media)
        if wt.text:
            txt = wt.text
            if not btn:
                txt, btn = get_msg_button(wt.text)
            add_note(chat, wrd, txt, m, btn)
        else:
            add_note(chat, wrd, None, m, btn)
    else:
        txt = wt.text
        if not btn:
            txt, btn = get_msg_button(wt.text)
        add_note(chat, wrd, txt, None, btn)

    if is_enabled():
        ultroid_bot.add_handler(
            notes, events.NewMessage(func=lambda e: e.text and not e.media)
        )
    await e.eor(get_string("notes_2").format(wrd))


@ultroid_cmd(pattern="remnote( (.*)|$)", admins_only=True)
async def rm_n(e):
    wrd = (e.pattern_match.group(1).strip()).lower()
    chat = e.chat_id
    if not wrd:
        return await e.eor(get_string("notes_3"), time=5)
    if wrd.startswith("#"):
        wrd = wrd.replace("#", "")
    rem_note(int(chat), wrd)
    await e.eor(f"Done Note: `#{wrd}` Removed.")


@ultroid_cmd(pattern="listnote$", admins_only=True)
async def lsnote(e):
    if x := list_note(e.chat_id):
        sd = "Notes Found In This Chats Are\n\n"
        return await e.eor(sd + x)
    await e.eor(get_string("notes_5"))


if udB.get_key("NOTE"):
    ultroid_bot.add_handler(
        notes, events.NewMessage(func=lambda e: e.text and not e.media)
    )

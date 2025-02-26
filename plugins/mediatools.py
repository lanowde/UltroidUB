# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

"""
✘ Commands Available -

• `{i}mediainfo <reply to media>/<file path>/<url>`
   To get info about it.

• `{i}rotate <degree/angle> <reply to media>`
   Rotate any video/photo/media..
   Note: for video it should be in angle of 90s.
"""

import os
import time
from datetime import datetime as dt
from shlex import quote as shquote

from pyUltroid.fns.misc import rotate_image

from . import (
    Catbox,
    LOGS,
    TelegraphClient,
    bash,
    get_string,
    is_url_ok,
    mediainfo,
    osremove,
    tg_downloader,
    ultroid_cmd,
)

try:
    import cv2
except ImportError:
    LOGS.info("WARNING: 'cv2' not found!")
    cv2 = None


@ultroid_cmd(pattern="mediainfo( (.*)|$)")
async def mi(e):
    r = await e.get_reply_message()
    match = e.pattern_match.group(1).strip()
    msg = await e.eor(f"`Loading Mediainfo...`")
    extra = ""

    if r and r.media:
        xx = mediainfo(r.media)
        murl = r.media.stringify()
        url = await TelegraphClient.create_page(
            title="Mediainfo",
            html_content=f"<pre>{murl}</pre>",
        )
        extra = f"[{xx}]({url})"
        naam, _ = await tg_downloader(
            media=r,
            event=msg,
            show_progress=True,
        )
    elif match and (
        os.path.isfile(match)
        or (match.startswith("https://") and await is_url_ok(match))
    ):
        naam, xx = match, "file"
    else:
        return await msg.eor(get_string("cvt_3"), time=5)

    out, er = await bash(f"mediainfo {shquote(naam)}")
    if er:
        LOGS.info(er)
        out = extra or str(er)
        if not match:
            osremove(naam)
        return await msg.edit(out, link_preview=False)

    makehtml = ""
    if naam.endswith((".jpg", ".png")):
        if os.path.exists(naam):
            med = await Catbox(naam)
        else:
            med = match
        makehtml += f"<img src='{med}'><br>"
    for line in out.split("\n"):
        line = line.strip()
        if not line:
            makehtml += "<br>"
        elif ":" not in line:
            makehtml += f"<h3>{line}</h3>"
        else:
            makehtml += f"<p>{line}</p>"
    try:
        urll = await TelegraphClient.create_page(
            title="Mediainfo", html_content=makehtml
        )
    except Exception as er:
        LOGS.exception(er)
        return await msg.edit(f"**Error:** `{er}`")
    await msg.edit(f"{extra} \n\n[{get_string('mdi_1')}]({urll})", link_preview=False)
    if not match:
        osremove(naam)


@ultroid_cmd(pattern="rotate( (.*)|$)")
async def rotate_(ult):
    match = ult.pattern_match.group(1).strip()
    if not ult.is_reply:
        return await ult.eor("`Reply to a media...`")
    if match:
        try:
            match = int(match)
        except ValueError:
            match = None
    if not match:
        return await ult.eor("`Please provide a valid angle to rotate media..`")
    reply = await ult.get_reply_message()
    msg = await ult.eor(get_string("com_1"))
    photo = reply.game.photo if reply.game else None
    if photo or reply.photo or reply.sticker:
        media = await ult.client.download_media(photo or reply)
        img = cv2.imread(media)
        new_ = rotate_image(img, match)
        file = "ult.png"
        cv2.imwrite(file, new_)
    elif reply.video:
        media = await reply.download_media()
        file = f"{media}.mp4"
        await bash(
            f"ffmpeg -i {shquote(media)} -c copy -metadata:s:v:0 rotate={match} {shquote(file)} -y"
        )
    else:
        return await msg.edit("`Unsupported Media..\nReply to Photo/Video`")
    if os.path.exists(file):
        await ult.client.send_file(
            ult.chat_id, file=file, video_note=bool(reply.video_note), reply_to=reply.id
        )
    os.remove(media)
    await msg.try_delete()

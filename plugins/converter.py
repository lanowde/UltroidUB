# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

from . import get_help

__doc__ = get_help("help_converter")

import asyncio
import os
import time
from io import BytesIO
from shlex import quote

from . import LOGS

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    LOGS.info(f"{__file__}: PIL  not Installed.")
    Image = None

from pyUltroid.custom._transfer import pyroDL, pyroUL
from . import (
    Catbox,
    ULTConfig,
    asyncwrite,
    asyncread,
    bash,
    check_filename,
    cleargif,
    con,
    get_paste,
    get_string,
    tg_downloader,
    udB,
    ultroid_cmd,
    unix_parser,
)

opn = []


@ultroid_cmd(
    pattern="thumbnail$",
)
async def set_thumbnail(e):
    r = await e.get_reply_message()
    if r.photo:
        dl = await r.download_media()
    elif r.document and r.document.thumbs:
        dl = await r.download_media(thumb=-1)
    else:
        return await e.eor("`Reply to Photo or media with thumb...`")

    nn = await Catbox(dl)
    os.remove(dl)
    udB.set_key("CUSTOM_THUMBNAIL", nn)
    await bash(f"wget '{nn}' -O resources/extras/ultroid.jpg")
    await e.eor(get_string("cvt_6").format(nn), link_preview=False)


@ultroid_cmd(
    pattern="rename( (.*)|$)",
)
async def imak(event):
    reply = await event.get_reply_message()
    if not (reply and reply.media):
        return await event.eor(get_string("cvt_1"))

    args = event.pattern_match.group(2)
    args = unix_parser(args or "")
    if not (inp := args.args):
        return await event.eor(get_string("cvt_2"))

    xx = await event.eor(get_string("com_1"))
    file, _ = await tg_downloader(media=reply, event=xx, show_progress=True)
    inp = check_filename(inp)
    await bash(f"mv -f {quote(file)} {quote(inp)}")
    if not os.path.exists(inp) or (os.path.exists(inp) and not os.path.getsize(inp)):
        os.rename(file, inp)
    ul = pyroUL(event=xx, _path=inp)
    await ul.upload(
        delete_file=True,
        reply_to=reply.id,
        auto_edit=False,
        _log=False,
        **args.kwargs,
    )
    await xx.delete()


conv_keys = {
    "img": "png",
    "sticker": "webp",
    "webp": "webp",
    "image": "png",
    "webm": "webm",
    "gif": "gif",
    "json": "json",
    "tgs": "tgs",
}


@ultroid_cmd(
    pattern="convert( (.*)|$)",
)
async def uconverter(event):
    xx = await event.eor(get_string("com_1"))
    a = await event.get_reply_message()
    if not (a and a.media):
        return await xx.eor("`Reply to Photo or media with thumb...`", time=6)

    input_ = event.pattern_match.group(2)
    b, _ = await tg_downloader(media=a, event=xx, show_progress=True)
    if not b and (a.document and a.document.thumbs):
        b = await a.download_media(thumb=-1)
    if not b:
        return await xx.edit(get_string("cvt_3"))
    try:
        convert = conv_keys[input_]
    except KeyError:
        return await xx.edit(get_string("sts_3").format("gif/img/sticker/webm"))
    file = await con.convert(b, outname="ultroid", convert_to=convert)
    if file:
        x = await a.reply(file=file)
        os.remove(file)
        await cleargif(x)
    else:
        return await xx.edit("`Failed to convert..`")
    await xx.delete()


@ultroid_cmd(
    pattern="(d)?doc( (.*)|$)",
)
async def d_doc(event):
    input_str = event.pattern_match.group(3)
    if not (input_str and event.reply_to):
        return await event.eor(get_string("cvt_1"), time=5)
    xx = await event.eor(get_string("com_1"))
    a = await event.get_reply_message()
    if not a.text:
        return await xx.edit(get_string("ex_1"))

    await xx.edit(f"**Packing into** `{input_str}..`")
    await asyncio.sleep(0.6)
    if event.pattern_match.group(1):
        out = check_filename(input_str)
        await asyncwrite(out, a.message, mode="w+")
        return await xx.edit(f"**Successfully Saved as** `{out}`")

    with BytesIO(a.message.encode()) as out:
        out.name = input_str
        await event.client.send_file(
            event.chat_id,
            file=out,
            caption=f"`{input_str}`",
            thumb=ULTConfig.thumb,
            reply_to=event.reply_to_msg_id,
        )
    await xx.delete()


@ultroid_cmd(
    pattern="open( (.*)|$)",
)
async def open_file_on_tg(event):
    a = await event.get_reply_message()
    b = event.pattern_match.group(2)
    if not ((a and a.media) or (b and os.path.exists(b))):
        return await event.eor(get_string("cvt_7"), time=5)

    xx = await event.eor(get_string("com_1"))
    rem = None
    if not b:
        b, _ = await tg_downloader(media=a, event=xx, show_progress=False)
        rem = True
    try:
        d = await asyncread(b)
    except UnicodeDecodeError:
        return await xx.eor(get_string("cvt_8"), time=5)
    try:
        await xx.edit(f"<code>{d}</code>", parse_mode="html")
    except BaseException:
        ext = (a.file.ext or ".txt")[1:]
        what, key = await get_paste(d, extension=ext)
        await xx.edit(
            f"**MESSAGE EXCEEDS TELEGRAM LIMITS**\n\nSo Pasted It On [SPACEBIN](https://spaceb.in/{key})"
        )
    if rem:
        os.remove(b)

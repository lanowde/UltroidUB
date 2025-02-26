# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

from . import get_help

__doc__ = get_help("help_downloadupload")

import asyncio
import glob
import os
import time
from pathlib import Path

# from aiohttp.client_exceptions import InvalidURL
from telethon.errors.rpcerrorlist import MessageNotModifiedError
from telethon.tl.types import DocumentAttributeAudio, DocumentAttributeVideo

from pyUltroid.fns.helper import time_formatter
from pyUltroid.fns.tools import get_chat_and_msgid
from pyUltroid.custom._transfer import gen_video_thumb, gen_audio_thumb, pyroDL, pyroUL

from . import (
    LOGS,
    ULTConfig,
    check_filename,
    cleargif,
    fast_download,
    gen_mediainfo,
    get_all_files,
    get_string,
    get_tg_filename,
    progress,
    string_is_url,
    time_formatter,
    tg_downloader,
    ultroid_cmd,
    unix_parser,
)


async def _url_downloader(link, filename, event):
    start = time.time()
    link = link if len(link) < 201 else (link[:200] + " ...")
    task = lambda d, t: asyncio.create_task(
        progress(d, t, event, start, f"Downloading from {link[:64]}...")
    )
    return await fast_download(link, filename, progress_callback=task)


@ultroid_cmd(
    pattern="download( (.*)|$)",
)
async def downlomder(event):
    matched = event.pattern_match.group(2)
    msg = await event.eor(get_string("udl_4"))
    if not matched:
        return await msg.eor(get_string("udl_5"), time=5)
    if "|" in matched:
        link, filename = map(lambda i: i.strip(), matched.split("|"))
    else:
        link, filename = matched, None
    try:
        filename, d = await _url_downloader(link, filename, msg)
    except Exception as exc:
        LOGS.exception(exc)
        return await msg.edit(f"**Error in URL download:** \n`{exc}`")

    await msg.edit(f"Downloaded to `{filename}` \nin {time_formatter(d * 1000)}.")


@ultroid_cmd(
    pattern="dl( (.*)|$)",
)
async def download(event):
    match = event.pattern_match.group(2)
    if match and "t.me/" in match:
        chat, msg = get_chat_and_msgid(match)
        if not (chat and msg):
            return await event.eor(get_string("gms_1"))
        match = ""
        ok = await event.client.get_messages(chat, ids=msg)
    elif event.reply_to:
        ok = await event.get_reply_message()
        if not (ok and ok.media):
            return await event.eor(get_string("udl_1"), time=5)
    else:
        return await event.eor(get_string("cvt_3"), time=8)

    xx = await event.eor(get_string("com_1"))
    filename = (
        (match if "resources/downloads/" in match else f"resources/downloads/{match}")
        if match
        else None
    )
    path, t_time = await tg_downloader(
        media=ok,
        event=xx,
        filename=filename,
        show_progress=True,
    )
    t = time_formatter(t_time * 1000)
    await xx.edit(get_string("udl_2").format(path, t))


@ultroid_cmd(pattern="xdl(?: |$)(.*)")
async def pyro_dl(event):
    reply = await event.get_reply_message()
    if not (reply and reply.media):
        return await event.eor("`Reply to a file to download..`", time=8)

    xx = await event.eor(get_string("com_1"))
    args = event.pattern_match.group(1)
    args = unix_parser(args or "")
    dlx = pyroDL(event=xx, source=reply)
    await dlx.download(**args.kwargs)


async def get_metadata(path, gen_thumb):
    data = await gen_mediainfo(path)
    if not data:
        return None, None

    thumb, attributes = None, None
    if data.get("type") == "audio":
        if gen_thumb:
            thumb = await gen_audio_thumb(path)
        attributes = [
            DocumentAttributeAudio(
                duration=data.get("duration", 0),
                title=data.get("title", Path(path).stem),
                performer=data.get("performer"),
            )
        ]
    elif data.get("type") == "video":
        if gen_thumb:
            thumb = await gen_video_thumb(
                path, data.get("duration"), data.get("type") == "gif"
            )
        attributes = [
            DocumentAttributeVideo(
                duration=data.get("duration", 0),
                w=data.get("width", 512),
                h=data.get("height", 512),
                supports_streaming=True,
            )
        ]

    return thumb, attributes


@ultroid_cmd(
    pattern="ul( (.*)|$)",
)
async def uploamder(event):
    msg = await event.eor(get_string("com_1"))
    args = event.pattern_match.group(2)
    args = unix_parser(args or "")
    if not (match := args.args):
        return await event.eor("`What Should I Upload !??`")

    if any(i in match.lower() for i in (".env", ".session")):
        return await event.reply("`You can't do this...`")

    stream, force_doc, delete, thumb = (
        False,
        True,
        False,
        ULTConfig.thumb,
    )
    if args.kwargs.get("-stream"):
        stream = True
        force_doc = False
    if args.kwargs.get("-delete"):  # --delete
        delete = True
    if args.kwargs.get("-no-thumb"):
        thumb = None

    if match.endswith("/"):
        match += "*"
    results = glob.glob(match)
    if not results and os.path.exists(match):
        results = [match]
    if not results:
        try:
            await event.reply(file=match)
            return await event.try_delete()
        except Exception as er:
            LOGS.exception(er)
        return await msg.eor(get_string("ls1"))

    for result in results:
        _thumb = thumb
        if os.path.isdir(result):
            c, s = 0, 0
            for files in get_all_files(result):
                if os.path.getsize(files) == 0:
                    c += 1
                    await msg.edit(f"`file size is 0B..`\n`{files}`")
                    await asyncio.sleep(6)
                    continue
                attributes = None
                if stream:
                    _thumb, attributes = await get_metadata(files, thumb)
                try:
                    file, _ = await event.client.fast_uploader(
                        files,
                        show_progress=True,
                        event=msg,
                        to_delete=delete,
                    )
                    y = await event.client.send_file(
                        event.chat_id,
                        file,
                        supports_streaming=stream,
                        force_document=force_doc,
                        thumb=_thumb,
                        attributes=attributes,
                        caption=f"`Uploaded` `{files}` `in {time_formatter(_ * 1000)}`",
                        reply_to=event.reply_to_msg_id or event,
                    )
                    s += 1
                    await cleargif(y)
                except (ValueError, IsADirectoryError):
                    c += 1
                finally:
                    await asyncio.sleep(6)
            break
        if os.path.getsize(result) == 0:
            return await msg.edit(f"`file size is 0B..`\n`{result}`")
        attributes = None
        if stream:
            _thumb, attributes = await get_metadata(result, thumb)
        file, _ = await event.client.fast_uploader(
            result,
            show_progress=True,
            event=msg,
            to_delete=delete,
        )
        y = await event.client.send_file(
            event.chat_id,
            file,
            supports_streaming=stream,
            force_document=force_doc,
            thumb=_thumb,
            attributes=attributes,
            caption=f"`Uploaded` `{result}` `in {time_formatter(_ * 1000)}`",
        )
        await cleargif(y)
    await msg.try_delete()


@ultroid_cmd(pattern="xul(?: |$)(.*)")
async def pyro_ul(e):
    msg = await e.eor(get_string("com_1"))
    args = e.pattern_match.group(1)
    args = unix_parser(args or "")
    ul_path, kwargs = args.args, args.kwargs
    if not ul_path or any(i in ul_path.lower() for i in (".env", ".session")):
        return await msg.edit("`Give some path/URL..`")

    if string_is_url(ul_path):
        await msg.edit("`Starting URL Download..`")
        await asyncio.sleep(2)
        ul_path, _ = await _url_downloader(ul_path, None, msg)
        await msg.edit(f"`Downloaded to {ul_path} \nUploading now..`")
        await asyncio.sleep(2)

    ulx = pyroUL(event=msg, _path=ul_path)
    await ulx.upload(**kwargs)

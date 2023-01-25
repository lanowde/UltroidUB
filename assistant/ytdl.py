# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

import os
import re

try:
    from PIL import Image
except ImportError:
    Image = None

from telethon import Button
from telethon.errors.rpcerrorlist import FilePartLengthInvalidError, MediaEmptyError
from telethon.tl.types import DocumentAttributeAudio, DocumentAttributeVideo
from telethon.tl.types import InputWebDocument as wb

from pyUltroid.fns.helper import (
    bash,
    fast_download,
    humanbytes,
    numerize,
    time_formatter,
)
from pyUltroid.fns.ytdl import dler, get_buttons, get_formats
from pyUltroid.fns.tools import check_filename

from . import LOGS, asst, callback, in_pattern, udB

try:
    from youtubesearchpython import VideosSearch
except ImportError:
    LOGS.info("'youtubesearchpython' not installed!")
    VideosSearch = None


ytt = "https://graph.org/file/afd04510c13914a06dd03.jpg"
_yt_base_url = "https://www.youtube.com/watch?v="
BACK_BUTTON = {}


@in_pattern("yt", owner=True)
async def _(event):
    try:
        string = event.text.split(" ", maxsplit=1)[1]
    except IndexError:
        fuk = event.builder.article(
            title="Search Something",
            thumb=wb(ytt, 0, "image/jpeg", []),
            text="**YᴏᴜTᴜʙᴇ Sᴇᴀʀᴄʜ**\n\nYou didn't search anything",
            buttons=Button.switch_inline(
                "Sᴇᴀʀᴄʜ Aɢᴀɪɴ",
                query="yt ",
                same_peer=True,
            ),
        )
        await event.answer([fuk])
        return
    results = []
    search = VideosSearch(string, limit=50)
    nub = search.result()
    nibba = nub["result"]
    for v in nibba:
        ids = v["id"]
        link = _yt_base_url + ids
        title = v["title"]
        duration = v["duration"]
        views = v["viewCount"]["short"]
        publisher = v["channel"]["name"]
        published_on = v["publishedTime"]
        description = (
            v["descriptionSnippet"][0]["text"]
            if v.get("descriptionSnippet")
            and len(v["descriptionSnippet"][0]["text"]) < 500
            else "None"
        )
        thumb = f"https://i.ytimg.com/vi/{ids}/hqdefault.jpg"
        text = f"**Title: [{title}]({link})**\n\n"
        text += f"`Description: {description}\n\n"
        text += f"「 Duration: {duration} 」\n"
        text += f"「 Views: {views} 」\n"
        text += f"「 Publisher: {publisher} 」\n"
        text += f"「 Published on: {published_on} 」`"
        desc = f"{title}\n{duration}"
        file = wb(thumb, 0, "image/jpeg", [])
        buttons = [
            [
                Button.inline("Audio", data=f"ytdl:audio:{ids}"),
                Button.inline("Video", data=f"ytdl:video:{ids}"),
            ],
            [
                Button.switch_inline(
                    "Sᴇᴀʀᴄʜ Aɢᴀɪɴ",
                    query="yt ",
                    same_peer=True,
                ),
                Button.switch_inline(
                    "Sʜᴀʀᴇ",
                    query=f"yt {string}",
                    same_peer=False,
                ),
            ],
        ]
        BACK_BUTTON.update({ids: {"text": text, "buttons": buttons}})
        results.append(
            await event.builder.article(
                type="photo",
                title=title,
                description=desc,
                thumb=file,
                content=file,
                text=text,
                include_media=True,
                buttons=buttons,
            ),
        )
    await event.answer(results[:50])


@callback(
    re.compile(
        "ytdl:(.*)",
    ),
    owner=True,
)
async def _(e):
    _e = e.pattern_match.group(1).strip().decode("UTF-8")
    _lets_split = _e.split(":")
    _ytdl_data = await dler(e, _yt_base_url + _lets_split[1])
    _data = get_formats(_lets_split[0], _lets_split[1], _ytdl_data)
    _buttons = get_buttons(_data)
    _text = (
        "`Select Your Format.`"
        if _buttons
        else "`Error downloading from YouTube.\nTry Restarting your bot.`"
    )

    await e.edit(_text, buttons=_buttons)


@callback(
    re.compile(
        "ytdownload:(.*)",
    ),
    owner=True,
)
async def _(event):
    from asyncio import sleep

    url = event.pattern_match.group(1).strip().decode("UTF-8")
    lets_split = url.split(":")
    vid_id = lets_split[2]
    link = _yt_base_url + vid_id
    format = lets_split[1]
    try:
        ext = lets_split[3]
    except IndexError:
        ext = "mp3"
    if lets_split[0] == "audio":
        opts = {
            "format": "bestaudio",
            "addmetadata": True,
            "key": "FFmpegMetadata",
            "prefer_ffmpeg": True,
            "geo_bypass": True,
            "outtmpl": f"%(id)s.{ext}",
            "logtostderr": False,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": ext,
                    "preferredquality": format,
                },
                {"key": "FFmpegMetadata"},
            ],
        }

        ytdl_data = await dler(event, link, opts, True)
        title = ytdl_data["title"]
        if ytdl_data.get("artist"):
            artist = ytdl_data["artist"]
        elif ytdl_data.get("creator"):
            artist = ytdl_data["creator"]
        elif ytdl_data.get("channel"):
            artist = ytdl_data["channel"]
        views = numerize(ytdl_data.get("view_count")) or 0
        thumb, _ = await fast_download(ytdl_data["thumbnail"], filename=f"{vid_id}.jpg")

        likes = numerize(ytdl_data.get("like_count")) or 0
        duration = ytdl_data.get("duration") or 0
        description = ytdl_data["description"][:120]
        description = description or "None"
        if not (filepath := ytHelper(vid_id, title)):
            return LOGS.error(f"YTDL ERROR: file not found: {vid_id}")
        size = os.path.getsize(filepath)
        from pyUltroid.custom._transfer import pyroUL

        ytaud = pyroUL(event=event, _path=filepath)
        yt_file = await ytaud.upload(
            thumb=thumb,
            auto_edit=False,
            return_obj=True,
            caption=filepath,
            delete_file=True,
            progress_text=f"`Uploading {filepath} ...`",
        )
    elif lets_split[0] == "video":
        opts = {
            "format": str(format),
            "addmetadata": True,
            "key": "FFmpegMetadata",
            "prefer_ffmpeg": True,
            "geo_bypass": True,
            "outtmpl": f"%(id)s.{ext}",
            "logtostderr": False,
            "postprocessors": [{"key": "FFmpegMetadata"}],
        }

        ytdl_data = await dler(event, link, opts, True)
        title = ytdl_data["title"]
        if ytdl_data.get("artist"):
            artist = ytdl_data["artist"]
        elif ytdl_data.get("creator"):
            artist = ytdl_data["creator"]
        elif ytdl_data.get("channel"):
            artist = ytdl_data["channel"]
        views = numerize(ytdl_data.get("view_count")) or 0
        thumb, _ = await fast_download(ytdl_data["thumbnail"], filename=f"{vid_id}.jpg")

        try:
            Image.open(thumb).save(thumb, "JPEG")
        except Exception as er:
            LOGS.exception(er)
            thumb = None
        description = ytdl_data["description"][:120]
        likes = numerize(ytdl_data.get("like_count")) or 0
        hi, wi = ytdl_data.get("height") or 720, ytdl_data.get("width") or 1280
        duration = ytdl_data.get("duration") or 0
        if not (filepath := ytHelper(vid_id, title)):
            return LOGS.error(f"YTDL ERROR: file not found: {vid_id}")
        size = os.path.getsize(filepath)
        from pyUltroid.custom._transfer import pyroUL

        ytvid = pyroUL(event=event, _path=filepath)
        yt_file = await ytvid.upload(
            thumb=thumb,
            auto_edit=False,
            return_obj=True,
            caption=filepath,
            delete_file=True,
            progress_text=f"`Uploading {filepath} ...`",
        )
    await sleep(1)
    description = description if description != "" else "None"
    text = f"**Title: [{title}]({_yt_base_url}{vid_id})**\n\n"
    text += f"`📝 Description: {description}\n\n"
    text += f"「 Duration: {time_formatter(int(duration)*1000)} 」\n"
    text += f"「 Artist: {artist} 」\n"
    text += f"「 Views: {views} 」\n"
    text += f"「 Likes: {likes} 」`"
    # text += f"「 Size: {humanbytes(size)} 」`"
    button = Button.switch_inline("Search More", query="yt ", same_peer=True)
    msg_to_edit = await asst.get_messages(yt_file.chat.id, ids=yt_file.id)
    await event.edit(text, file=msg_to_edit.media, buttons=button)


@callback(re.compile("ytdl_back:(.*)"), owner=True)
async def ytdl_back(event):
    id_ = event.data_match.group(1).decode("utf-8")
    if not BACK_BUTTON.get(id_):
        return await event.answer("Query Expired! Search again 🔍")
    await event.edit(**BACK_BUTTON[id_])


def ytHelper(yt_id, title):
    exts = (
        ".webm",
        ".mkv",
        ".mp4",
        ".3gp",
        ".mp3",
        ".m4a",
        ".flv",
        ".m4v",
        ".aac",
        ".opus",
        ".ogg",
        ".wav",
    )
    for file in os.listdir("."):
        for ext in exts:
            if file.startswith(yt_id) and file.lower().endswith(ext):
                fn = check_filename(title + "." + file.split(".", maxsplit=1)[1])
                os.rename(file, fn)
                return fn

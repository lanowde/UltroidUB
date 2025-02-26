# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

import asyncio
import os
import re
import time
from pathlib import Path

from telethon import Button
from telethon.tl.types import InputWebDocument

from pyUltroid.fns.helper import download_file, numerize
from pyUltroid.custom._transfer import pyroUL
from pyUltroid.fns.ytdl import extract_info, dler, get_buttons, get_formats
from pyUltroid.custom.commons import (
    bash,
    check_filename,
    osremove,
    humanbytes,
    time_formatter,
)

from . import LOGS, asst, callback, in_pattern, udB

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from youtubesearchpython.__future__ import VideosSearch
except ImportError:
    LOGS.info("'youtubesearchpython' is not installed. Some plugins will not work!")
    VideosSearch = None


ytt = "https://graph.org/file/afd04510c13914a06dd03.jpg"
_yt_base_url = "https://www.youtube.com/watch?v="
BACK_BUTTON = {}


@in_pattern("yt", owner=True)
async def inline_yt(event):
    try:
        string = event.text.split(" ", maxsplit=1)[1]
    except IndexError:
        fuk = event.builder.article(
            title="Search Something",
            thumb=InputWebDocument(ytt, 0, "image/jpeg", []),
            text="**YᴏᴜTᴜʙᴇ Sᴇᴀʀᴄʜ**\n\nYou didn't search anything",
            buttons=Button.switch_inline(
                "Sᴇᴀʀᴄʜ Aɢᴀɪɴ",
                query="yt ",
                same_peer=True,
            ),
        )
        return await event.answer([fuk])

    results = []
    obj = VideosSearch(string, limit=30)
    nub = await obj.next()
    for v in nub["result"]:
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
        text += f"`Description: {description}...\n\n"
        text += f"「 Duration: {duration} 」\n"
        text += f"「 Views: {views} 」\n"
        text += f"「 Publisher: {publisher} 」\n"
        text += f"「 Published on: {published_on} 」`"
        desc = f"{title}\n{duration}"
        file = InputWebDocument(thumb, 0, "image/jpeg", [])
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
        BACK_BUTTON[ids] = {"text": text, "buttons": buttons}
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
    await event.answer(results)


@callback(
    re.compile(
        "ytdl:(.*)",
    ),
    owner=True,
)
async def inline_ytdl_format(e):
    _e = e.pattern_match.group(1).strip().decode("UTF-8")
    _lets_split = _e.split(":")
    _ytdl_data = await extract_info(_yt_base_url + _lets_split[1])
    _data = get_formats(_lets_split[0], _lets_split[1], _ytdl_data)
    _buttons = get_buttons(_data)
    _text = (
        "`Select Your Format.`"
        if _buttons
        else "`Error fetching data from YouTube.\nTry Restarting your bot.`"
    )
    await e.edit(_text, buttons=_buttons)


@callback(
    re.compile(
        "ytdownload:(.*)",
    ),
    owner=True,
)
async def inline_ytdownload(event):
    url = event.pattern_match.group(1).strip().decode("UTF-8")
    lets_split = url.split(":")
    vid_id = lets_split[2]
    link = _yt_base_url + vid_id
    format = lets_split[1]
    try:
        ext = lets_split[3]
    except IndexError:
        ext = "mp3"

    folder = f"resources/temp/{time.time()}"
    ytdl_data = await extract_info(link)
    if lets_split[0] == "audio":
        opts = {
            "format": "bestaudio",
            "folder": folder,
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": ext,
                    "preferredquality": format,
                },
                {"key": "FFmpegMetadata"},
            ],
        }

        resp, folder = await dler(event, link, **opts)
        if resp != 0:
            LOGS.error(
                f"YTDL ERROR: Something went Wrong in downloading! - return code: {resp} - Check folder: {folder}"
            )
            return await event.edit(
                f"`Something went wrong while downloading this audio...`"
            )

        title = ytdl_data.get("title")
        filepath = None
        for pth in Path(folder).iterdir():
            if pth.stem.startswith(vid_id) and not pth.suffix.endswith(
                (".jpg", ".png", ".webp")
            ):
                new_stem = pth.stem.replace(f"{vid_id}-", f"{title}-")
                filepath = str(pth.rename(pth.parent / (new_stem + pth.suffix)))

        if not filepath:
            LOGS.error(
                f"YTDL ERROR: could not found audio file - {folder}/{title} - return code: {resp} - (possibly wrong format or server error)"
            )
            return await event.edit(
                "`Error: File could not be Downloaded! Try choosing different format..`"
            )

        if filepath.lower().endswith((".part", ".temp", ".tmp", ".ytdl")):
            osremove(filepath)
            LOGS.warning(
                f"YTDL ERROR: Corrupted file or Incomplete download: {folder}/{title} - response code: {resp} - audio file ending in .part or .temp..."
            )
            return await event.edit("`Error: Invalid audio format was Choosen...`")

        if not (thumbnail := ytdl_data.get("thumbnail")):
            for th in reversed(ytdl_data.get("thumbnails", [])):
                if th.get("url", "").endswith(".jpg"):
                    thumbnail = th["url"]
                    break

        if not thumbnail:
            thumbnail = f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"
        thumb, _ = await download_file(
            thumbnail,
            f"resources/temp/{title}.jpg",
        )

        if ytdl_data.get("artist"):
            artist = ytdl_data["artist"]
        elif ytdl_data.get("creator"):
            artist = ytdl_data["creator"]
        elif ytdl_data.get("channel"):
            artist = ytdl_data["channel"]
        views = numerize(ytdl_data.get("view_count")) or 0
        likes = numerize(ytdl_data.get("like_count")) or 0
        duration = ytdl_data.get("duration") or 0
        description = (ytdl_data["description"] or "None")[:120] + "..."

        yt_audio = pyroUL(event=event, _path=filepath)
        yt_file = await yt_audio.upload(
            thumb=thumb,
            auto_edit=False,
            return_obj=True,
            caption=filepath,
            delete_file=True,
            progress_text=f"`Uploading {Path(filepath).name} ...`",
        )

    elif lets_split[0] == "video":
        opts = {
            "format": str(format),
            "key": "FFmpegMetadata",
            "folder": folder,
            "noplaylist": True,
            "merge_output_format": "mkv/mp4/flv",
            "postprocessors": [{"key": "FFmpegMetadata"}],
        }

        resp, folder = await dler(event, link, **opts)
        if resp != 0:
            LOGS.error(
                f"YTDL ERROR: Something went Wrong in downloading! - return code: {resp} - Check folder: {folder}"
            )
            return await event.edit(
                f"`Something went wrong while downloading this video...`"
            )

        title = ytdl_data.get("title")
        filepath = None
        for pth in Path(folder).iterdir():
            if pth.stem.startswith(vid_id) and not pth.suffix.endswith(
                (".jpg", ".png", ".webp")
            ):
                new_stem = pth.stem.replace(f"{vid_id}-", f"{title}-")
                filepath = str(pth.rename(pth.parent / (new_stem + pth.suffix)))

        if not filepath:
            LOGS.warning(
                f"YTDL ERROR: could not found video file - {folder}/{title} - return code: {resp} - (possibly wrong format or server error)"
            )
            return await event.edit(
                "`Error: File could not be Downloaded! Try choosing different format..`"
            )

        if filepath.lower().endswith((".part", ".temp", ".tmp", ".ytdl")):
            osremove(filepath)
            LOGS.warning(
                f"YTDL ERROR: Corrupted file or Incomplete download: {folder}/{title} - return code: {resp} - file ending in .part or .temp..."
            )
            return await event.edit("`Error: Invalid video format was Choosen...`")

        if not (thumbnail := ytdl_data.get("thumbnail")):
            for th in reversed(ytdl_data.get("thumbnails", [])):
                if th.get("url", "").endswith(".jpg"):
                    thumbnail = th["url"]
                    break

        if not thumbnail:
            thumbnail = f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"
        thumb, _ = await download_file(
            thumbnail,
            f"resources/temp/{title}.jpg",
        )

        """
        try:
            if Image:
                Image.open(thumb).save(thumb, "JPEG")
        except Exception as er:
            LOGS.exception("YTDL Error in saving thumbnail..")
            thumb = None
        """

        if ytdl_data.get("artist"):
            artist = ytdl_data["artist"]
        elif ytdl_data.get("creator"):
            artist = ytdl_data["creator"]
        elif ytdl_data.get("channel"):
            artist = ytdl_data["channel"]

        views = numerize(ytdl_data.get("view_count")) or 0
        likes = numerize(ytdl_data.get("like_count")) or 0
        duration = ytdl_data.get("duration") or 0
        description = (ytdl_data["description"] or "None")[:100] + "..."
        # hi, wi = ytdl_data.get("height") or 720, ytdl_data.get("width") or 1280
        # size = os.path.getsize(filepath)

        yt_video = pyroUL(event=event, _path=filepath)
        yt_file = await yt_video.upload(
            thumb=thumb,
            auto_edit=False,
            return_obj=True,
            caption=filepath,
            delete_file=True,
            progress_text=f"`Uploading {Path(filepath).name} ...`",
        )

    text = f"**Title: [{title}]({_yt_base_url}{vid_id})**\n\n"
    text += f"`📝 Description: {description}\n\n"
    text += f"「 Duration: {time_formatter(int(duration) * 1000)} 」\n"
    text += f"「 Artist: {artist} 」\n"
    text += f"「 Views: {views} 」\n"
    text += f"「 Likes: {likes} 」`"
    # text += f"「 Size: {humanbytes(size)} 」`"

    button = Button.switch_inline("Search More", query="yt ", same_peer=True)
    msg_to_edit = await asst.get_messages(yt_file.chat.id, ids=yt_file.id)
    await asyncio.sleep(0.6)
    await event.edit(text, file=msg_to_edit.media, buttons=button)
    osremove(folder, folders=True)


@callback(re.compile("ytdl_back:(.*)"), owner=True)
async def cb_ytdl_back(event):
    id_ = event.data_match.group(1).decode("utf-8")
    if not BACK_BUTTON.get(id_):
        return await event.answer("Query Expired! Search again 🔍")
    await event.edit(**BACK_BUTTON[id_])

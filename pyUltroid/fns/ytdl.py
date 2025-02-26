# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

import asyncio
import os
import re
import time
from functools import partial
from pathlib import Path

from telethon import Button
from yt_dlp import YoutubeDL

from pyUltroid import LOGS, udB
from pyUltroid.startup._logger import logging
from pyUltroid.custom._transfer import pyroUL
from pyUltroid.custom._loop import tasks_db, run_async_task
from pyUltroid.custom.commons import (
    check_filename,
    humanbytes,
    osremove,
    run_async,
    time_formatter,
)
from .helper import download_file

try:
    from youtubesearchpython.__future__ import Playlist, VideosSearch
except ImportError:
    Playlist, VideosSearch = None, None


ytdl_logger = logging.getLogger("yt-dlp")


def _ytdl_options():
    opts = {}
    opts["quiet"] = True
    opts["nocheckcertificate"] = True
    file = "resources/extras/yt-cookies.txt"
    if Path(file).is_file():
        opts["cookiefile"] = file
    opts["username"] = udB.get_key("YT_USERNAME")
    opts["password"] = udB.get_key("YT_PASSWORD")
    opts["noprogress"] = True
    opts["overwrites"] = True
    opts["geo_bypass"] = True
    opts["addmetadata"] = True
    opts["prefer_ffmpeg"] = True
    opts["logger"] = ytdl_logger
    if _opts := udB.get_key("YTDL_ARGS"):
        if type(_opts) == dict:
            opts = _opts | opts

    return opts


_YT_PROGRESS = {}


def ytdl_progress(event, k):
    idx = f"{event.chat_id}_{event.id}"
    prev = _YT_PROGRESS.get(idx)
    now = time.time()
    if k["status"] != "downloading" or (prev and now - prev < 7):
        return

    try:
        total = k.get("total_bytes")
        downloaded = k.get("downloaded_bytes")
        eta = k.get("eta") or 0
        speed = k.get("speed") or 0
        if total and downloaded and "yt_progress_bar" not in tasks_db:
            text = (
                f"**YT Downloader**\n\n"
                + f"Status: `{humanbytes(downloaded)}/{humanbytes(total)}` "
                + f"`[{int(downloaded) * 100 / int(total):.2f}%]`\n"
                + f"Speed: `{humanbytes(speed)}/s`\n"
                + f"ETA: `{time_formatter(eta * 1000)}`"
            )
            run_async_task(event.edit, text, id="yt_progress_bar")
            _YT_PROGRESS[idx] = now
    except Exception as exc:
        LOGS.exception(exc)


async def get_yt_link(query):
    obj = VideosSearch(query, limit=1)
    search = await obj.next()
    try:
        return search["result"][0]["link"]
    except (IndexError, KeyError):
        return


@run_async
def ytdownload(url, opts, event):
    try:
        yt = YoutubeDL(opts)
        yt.add_progress_hook(partial(ytdl_progress, event))
        return yt.download(url)
    except Exception as ex:
        LOGS.exception(ex)


@run_async
def extract_info(url):
    return YoutubeDL(_ytdl_options()).extract_info(url=url, download=False)


# ---------------YouTube Downloader Inline---------------
# @New-Dev0 @buddhhu @1danish-00


def get_formats(type, id, data):
    if type == "audio":
        audio = []
        for _quality in ("64", "128", "256", "320"):
            audio.append(
                {
                    "ytid": id,
                    "type": "audio",
                    "id": _quality,
                    "quality": _quality + "KBPS",
                }
            )
        return audio
    if type == "video":
        video = []
        size = 0
        for vid in data["formats"]:
            if vid["format_id"] == "251":
                size += vid["filesize"] if vid.get("filesize") else 0
            if vid["vcodec"] != "none":
                _id = int(vid["format_id"])
                _quality = str(vid["width"]) + "×" + str(vid["height"])
                _size = size + (vid["filesize"] if vid.get("filesize") else 0)
                _ext = "mkv" if vid["ext"] == "webm" else "mp4"
                if _size < 2147483648:  # Telegram's Limit of 2GB
                    video.append(
                        {
                            "ytid": id,
                            "type": "video",
                            "id": str(_id) + "+251",
                            "quality": _quality,
                            "size": _size,
                            "ext": _ext,
                        }
                    )
        return video
    return []


def get_buttons(listt):
    id = listt[0]["ytid"]
    butts = [
        Button.inline(
            text=f"[{x['quality']}"
            + (f" {humanbytes(x['size'])}]" if x.get("size") else "]"),
            data=f"ytdownload:{x['type']}:{x['id']}:{x['ytid']}"
            + (f":{x['ext']}" if x.get("ext") else ""),
        )
        for x in listt
    ]
    buttons = list(zip(butts[::2], butts[1::2]))
    if len(butts) % 2 == 1:
        buttons.append((butts[-1],))
    buttons.append([Button.inline("« Back", f"ytdl_back:{id}")])
    return buttons


async def dler(event, url, **opts):
    await event.edit("`Getting Data...`")
    folder = opts.pop("folder", f"resources/temp/{time.time()}")
    opts = _ytdl_options() | opts
    opts["outtmpl"] = folder + "/" + "%(id)s-%(format)s.%(ext)s"
    re_code = await ytdownload(url, opts, event)
    _YT_PROGRESS.pop(f"{event.chat_id}_{event.id}", None)
    return (re_code, folder)


async def get_videos_link(url):
    to_return = []
    regex = re.search(r"\?list=([(\w+)\-]*)", url)
    if not regex:
        return to_return
    playlist_id = regex.group(1)
    videos = Playlist(playlist_id)
    while video.hasMoreVideos:
        vid = await videos.getNextVideos()
        link = re.search(r"\?v=([(\w+)\-]*)", vid["link"]).group(1)
        to_return.append(f"https://youtube.com/watch?v={link}")

    return to_return


async def download_yt(event, link, ytd):
    reply_to = event.reply_to_msg_id or event
    code, folder = await dler(event, link, **ytd)
    if code != 0:
        return LOGS.error(
            f"Something went Wrong in YT downloader! return code: {code} - Check folder: {folder}"
        )

    try:
        info = await extract_info(link)
        if not info:
            LOGS.error(
                f"Something went Wrong while extracting info from YT! {link} - return code: {code}"
            )
    except Exception as exc:
        await event.edit(f"{type(exc)}: {exc}")
        return

    if info.get("_type") == "playlist":
        total = info["playlist_count"]
        for num, file in enumerate(info["entries"], start=1):
            vid_id = file["id"]
            title = file["title"]
            filepath = None
            for pth in Path(folder).iterdir():
                if pth.stem.startswith(vid_id) and not pth.suffix.endswith(
                    (".jpg", ".png", ".webp")
                ):
                    new_stem = pth.stem.replace(f"{vid_id}-", f"{title}-")
                    filepath = str(pth.rename(pth.parent / (new_stem + pth.suffix)))

            if not filepath:
                LOGS.warning(
                    f"YTDL ERROR: file not found - {folder}/{title} - return code: {code}"
                )
                continue

            if filepath.lower().endswith((".part", ".temp", ".tmp", ".ytdl")):
                # osremove(filepath)
                LOGS.warning(
                    f"YTDL Corrupted or Incomplete Download: {folder}/{title} - return code: {code} - file name ending in .part or .temp"
                )
                await asyncio.sleep(3)
                continue

            if not (thumbnail := file.get("thumbnail")):
                for th in reversed(file.get("thumbnails", [])):
                    if th.get("url", "").endswith(".jpg"):
                        thumbnail = th["url"]
                        break

            if not thumbnail:
                thumbnail = f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"
            thumb, _ = await download_file(
                thumbnail,
                f"resources/temp/{title}.jpg",
            )
            from_ = info["extractor"].split(":")[0]
            caption = f"`[{num}/{total}] {title}`\n\n`from {from_}`"
            ulx = pyroUL(event=event, _path=filepath)
            await ulx.upload(
                thumb=thumb,
                to_delete=True,
                caption=caption,
                auto_edit=False,
                delete_file=True,
                reply_to=reply_to,
            )
            await event.edit(f"`Uploaded {title}!`")
            await asyncio.sleep(5)

        osremove(folder, folders=True)
        return await event.try_delete()

    # single file
    title = info["title"]
    vid_id = info["id"]
    filepath = None
    for pth in Path(folder).iterdir():
        if pth.stem.startswith(vid_id) and not pth.suffix.endswith(
            (".jpg", ".png", ".webp")
        ):
            new_stem = pth.stem.replace(f"{vid_id}-", f"{title}-")
            filepath = str(pth.rename(pth.parent / (new_stem + pth.suffix)))

    if not filepath:
        return LOGS.warning(
            f"YTDL ERROR: file not found - {folder}/{title} - return code: {code}"
        )

    if filepath.lower().endswith((".part", ".temp", ".tmp", ".ytdl")):
        # osremove(filepath)
        return LOGS.warning(
            f"YTDL Corrupted or Incomplete Download: {folder}/{title} - return code: {code} - file name ending in .part or .temp"
        )

    if not (thumbnail := info.get("thumbnail")):
        for th in reversed(info.get("thumbnails", [])):
            if th.get("url", "").endswith(".jpg"):
                thumbnail = th["url"]
                break

    if not thumbnail:
        thumbnail = f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"
    thumb, _ = await download_file(
        thumbnail,
        f"resources/temp/{title}.jpg",
    )
    ulx = pyroUL(event=event, _path=filepath)
    await ulx.upload(
        thumb=thumb,
        to_delete=True,
        auto_edit=False,
        reply_to=reply_to,
        delete_file=True,
        caption=f"`{title}`",
    )
    # osremove(folder, folders=True)
    await event.try_delete()


__all__ = (
    "dler",
    "download_yt",
    "extract_info",
    "get_buttons",
    "get_formats",
    "get_videos_link",
    "get_yt_link",
    "ytdl_progress",
    "ytdownload",
)

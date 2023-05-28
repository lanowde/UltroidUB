# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

"""
✘ Commands Available -

• `{i}yta <(youtube/any) link>`
   Download audio from the link.

• `{i}ytv <(youtube/any) link>`
   Download video  from the link.

• `{i}ytsa <(youtube) search query>`
   Search and download audio from youtube.

• `{i}ytsv <(youtube) search query>`
   Search and download video from youtube.
"""

import asyncio
from urllib.parse import urlparse

from pyUltroid.fns.ytdl import download_yt, get_yt_link

from . import get_string, ultroid_cmd


@ultroid_cmd(
    pattern="yt(a|v|sa|sv) ?(.*)",
)
async def download_from_youtube_(event):
    def is_valid_url(url):
        result = urlparse(url)
        return all([result.scheme, result.netloc])

    ytd = {"nocheckcertificate": True}
    opt = event.pattern_match.group(1).strip()
    xx = await event.eor(get_string("com_1"))
    if opt == "a":
        ytd["format"] = "bestaudio"
        ytd["outtmpl"] = "%(id)s"
        url = event.pattern_match.group(2)
        if not url:
            return await xx.eor(get_string("youtube_1"))
        if not is_valid_url(url):
            return await xx.eor(get_string("youtube_2"))
    elif opt == "v":
        ytd["format"] = "best"
        ytd["outtmpl"] = "%(id)s"
        ytd["postprocessors"] = [{"key": "FFmpegMetadata"}]
        url = event.pattern_match.group(2)
        if not url:
            return await xx.eor(get_string("youtube_3"))
        if not is_valid_url(url):
            return await xx.eor(get_string("youtube_4"))
    elif opt == "sa":
        ytd["format"] = "bestaudio"
        ytd["outtmpl"] = "%(id)s"
        try:
            query = event.text.split(" ", 1)[1]
        except IndexError:
            return await xx.eor(get_string("youtube_5"))
        url = await asyncio.to_thread(get_yt_link, query)
        if not url:
            return await xx.edit(get_string("unspl_1"))
        await xx.eor(get_string("youtube_6"))
    elif opt == "sv":
        ytd["format"] = "best"
        ytd["outtmpl"] = "%(id)s"
        ytd["postprocessors"] = [{"key": "FFmpegMetadata"}]
        try:
            query = event.text.split(" ", 1)[1]
        except IndexError:
            return await xx.eor(get_string("youtube_7"))
        url = await asyncio.to_thread(get_yt_link, query)
        if not url:
            return await xx.edit(get_string("unspl_1"))
        await xx.eor(get_string("youtube_8"))
    else:
        return
    await download_yt(xx, url, ytd)

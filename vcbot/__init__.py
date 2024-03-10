# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

# ----------------------------------------------------------#
#                                                           #
#    _   _ _   _____ ____   ___ ___ ____   __     ______    #
#   | | | | | |_   _|  _ \ / _ \_ _|  _ \  \ \   / / ___|   #
#   | | | | |   | | | |_) | | | | || | | |  \ \ / / |       #
#   | |_| | |___| | |  _ <| |_| | || |_| |   \ V /| |___    #
#    \___/|_____|_| |_| \_\\___/___|____/     \_/  \____|   #
#                                                           #
# ----------------------------------------------------------#


import asyncio
import re
import traceback
from pathlib import Path
from time import time
from traceback import format_exc

from pytgcalls import GroupCallFactory
from pytgcalls.exceptions import GroupCallNotFoundError
from telethon import events
from telethon.tl import functions, types
from telethon.utils import get_display_name
from telethon.errors.rpcerrorlist import (
    ParticipantJoinMissingError,
    ChatSendMediaForbiddenError,
)

from strings import get_string
from pyUltroid import HNDLR, LOGS, asst, udB, vcClient
from pyUltroid._misc._decorators import compile_pattern
from pyUltroid.fns.helper import inline_mention, mediainfo, tg_downloader
from pyUltroid.custom.commons import (
    bash,
    check_filename,
    get_tg_filename,
    osremove,
    time_formatter,
)
from pyUltroid.fns.admins import admin_check
from pyUltroid.fns.tools import is_url_ok
from pyUltroid.fns.ytdl import get_videos_link
from pyUltroid._misc import owner_and_sudos, sudoers
from pyUltroid._misc._assistant import in_pattern
from pyUltroid._misc._wrappers import eod, eor
from pyUltroid.version import __version__ as UltVer

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None
    LOGS.error("VCBOT: 'yt-dlp' not found!")

try:
    from youtubesearchpython.__future__ import VideosSearch
except ImportError:
    VideosSearch = None


asstUserName = asst.me.username

LOG_CHANNEL = udB.get_key("LOG_CHANNEL")
VC_HNDLR = udB.get_key("VC_HNDLR") or HNDLR
VC_AUTH_GC = udB.get_key("VC_AUTH_GROUPS") or {}

CLIENTS = {}
ACTIVE_CALLS = []
VC_QUEUE = {}
MSGID_CACHE = {}
VIDEO_ON = {}


def VC_AUTHS():
    _vcsudos = udB.get_key("VC_SUDOS") or []
    return [int(a) for a in [*owner_and_sudos(), *_vcsudos]]


class Player:
    def __init__(self, chat, event=None, video=False):
        self._chat = chat
        self._current_chat = event.chat_id if event else LOG_CHANNEL
        self._video = video
        if CLIENTS.get(chat):
            self.group_call = CLIENTS[chat]
        else:
            _client = GroupCallFactory(
                vcClient,
                GroupCallFactory.MTPROTO_CLIENT_TYPE.TELETHON,
            )
            self.group_call = _client.get_group_call()
            CLIENTS.update({chat: self.group_call})

    async def make_vc_active(self):
        try:
            await vcClient(
                functions.phone.CreateGroupCallRequest(
                    self._chat, title="Ultroid VCBOT 🎧"
                )
            )
        except Exception as e:
            LOGS.exception(e)
            return False, e
        return True, None

    async def startCall(self):
        if VIDEO_ON:
            for chats in VIDEO_ON:
                await VIDEO_ON[chats].stop()
            VIDEO_ON.clear()
            await asyncio.sleep(2)
        if self._video:
            for chats in list(CLIENTS):
                if chats != self._chat:
                    await CLIENTS[chats].stop()
                    CLIENTS.pop(chats, None)
            VIDEO_ON.update({self._chat: self.group_call})
        if self._chat not in ACTIVE_CALLS:
            try:
                self.group_call.on_network_status_changed(self.on_network_changed)
                self.group_call.on_playout_ended(self.playout_ended_handler)
                await self.group_call.join(self._chat)
            except GroupCallNotFoundError as er:
                LOGS.exception(er)
                dn, err = await self.make_vc_active()
                if err:
                    return False, err
            except Exception as e:
                LOGS.exception(e)
                return False, e
        return True, None

    async def on_network_changed(self, call, is_connected):
        chat = self._chat
        if is_connected:
            if chat not in ACTIVE_CALLS:
                ACTIVE_CALLS.append(chat)
        elif chat in ACTIVE_CALLS:
            ACTIVE_CALLS.remove(chat)

    async def playout_ended_handler(self, call, source, mtype):
        osremove(source)
        await self.play_from_queue()

    async def play_from_queue(self):
        chat_id = self._chat
        if chat_id in VIDEO_ON:
            await self.group_call.stop_video()
            VIDEO_ON.pop(chat_id)
        try:
            (
                song,
                title,
                link,
                thumb,
                from_user,
                pos,
                dur,
                silent,
            ) = await get_from_queue(chat_id)
            try:
                await self.group_call.start_audio(song)
            except ParticipantJoinMissingError:
                await self.vc_joiner()
                await self.group_call.start_audio(song)

            if MSGID_CACHE.get(chat_id):
                await MSGID_CACHE[chat_id].try_delete()
                MSGID_CACHE.pop(chat_id, None)
            if not silent:
                text = "<strong>🎧 Now playing #{}: <a href={}>{}</a>\n⏰ Duration:</strong> <code>{}</code>\n👤 <strong>Requested by:</strong> {}".format(
                    pos, link, title, dur, from_user
                )
                try:
                    xx = await vcClient.send_message(
                        self._current_chat,
                        text[:1023],
                        file=thumb,
                        link_preview=False,
                        parse_mode="html",
                    )
                    MSGID_CACHE.update({chat_id: xx})
                except ChatSendMediaForbiddenError:
                    xx = await vcClient.send_message(
                        self._current_chat,
                        text,
                        link_preview=False,
                        parse_mode="html",
                    )
                    MSGID_CACHE.update({chat_id: xx})
                except BaseException as exc:
                    LOGS.exception(exc)
                    VC_QUEUE[chat_id].pop(pos, None)
            VC_QUEUE[chat_id].pop(pos, None)
            if not VC_QUEUE[chat_id]:
                VC_QUEUE.pop(chat_id, None)
            osremove(song, thumb)
        except (IndexError, KeyError):
            await self.group_call.stop()
            VC_QUEUE.pop(chat_id, None)
            CLIENTS.pop(self._chat, None)
            await vcClient.send_message(
                self._current_chat,
                f"• Successfully Left Vc : <code>{chat_id}</code> •",
                parse_mode="html",
            )
        except Exception as er:
            LOGS.exception(er)
            VC_QUEUE.pop(chat_id, None)
            await vcClient.send_message(
                self._current_chat,
                f"<strong>ERROR:</strong> <code>{format_exc()}</code>",
                parse_mode="html",
            )

    async def vc_joiner(self):
        chat_id = self._chat
        done, err = await self.startCall()

        if done:
            await vcClient.send_message(
                self._current_chat,
                f"• Joined VC in <code>{chat_id}</code>",
                parse_mode="html",
            )
            return True
        await vcClient.send_message(
            self._current_chat,
            f"<strong>ERROR while Joining Vc -</strong> <code>{chat_id}</code> :\n<code>{err}</code>",
            parse_mode="html",
        )
        return False


# --------------------------------------------------


def vc_asst(dec, **kwargs):
    def ult(func):
        kwargs["func"] = (
            lambda e: not e.is_private and not e.via_bot_id and not e.fwd_from
        )
        kwargs["pattern"] = compile_pattern(dec, VC_HNDLR)
        vc_auth = kwargs.get("vc_auth", True)
        if "vc_auth" in kwargs:
            del kwargs["vc_auth"]

        async def vc_handler(e):
            VCAUTH = list(VC_AUTH_GC.keys())
            if not (
                (e.out)
                or (e.sender_id in VC_AUTHS())
                or (vc_auth and e.chat_id in VCAUTH)
            ):
                return
            elif vc_auth and VC_AUTH_GC.get(e.chat_id):
                cha, adm = VC_AUTH_GC.get(e.chat_id), VC_AUTH_GC[e.chat_id]["admins"]
                if adm and not (await admin_check(e)):
                    return
            try:
                await func(e)
            except Exception as exc:
                LOGS.exception(exc)
                await asst.send_message(
                    LOG_CHANNEL,
                    f"VC Error - <code>{UltVer}</code>\n\n<code>{e.text}</code>\n\n<code>{format_exc()}</code>",
                    parse_mode="html",
                )

        vcClient.add_event_handler(
            vc_handler,
            events.NewMessage(**kwargs),
        )

    return ult


# --------------------------------------------------


def add_to_queue(
    chat_id, song, song_name, link, thumb, from_user, duration, silent=False
):
    try:
        n = sorted(list(VC_QUEUE[chat_id].keys()))
        play_at = n[-1] + 1
    except BaseException:
        play_at = 1
    stuff = {
        play_at: {
            "song": song,
            "title": song_name,
            "link": link,
            "thumb": thumb,
            "from_user": from_user,
            "duration": duration,
            "silent": silent,
        }
    }
    if VC_QUEUE.get(chat_id):
        VC_QUEUE[int(chat_id)].update(stuff)
    else:
        VC_QUEUE.update({chat_id: stuff})
    return VC_QUEUE[chat_id]


def list_queue(chat):
    if VC_QUEUE.get(chat):
        txt, n = "", 0
        for x in list(VC_QUEUE[chat].keys())[:18]:
            n += 1
            data = VC_QUEUE[chat][x]
            txt += f'<strong>{n}. <a href={data["link"]}>{data["title"]}</a> :</strong> <i>By: {data["from_user"]}</i>\n'
        txt += "\n\n....."
        return txt


async def get_from_queue(chat_id):
    play_this = list(VC_QUEUE[int(chat_id)].keys())[0]
    info = VC_QUEUE[int(chat_id)][play_this]
    song = info.get("song")
    title = info["title"]
    link = info["link"]
    thumb = info["thumb"]
    from_user = info["from_user"]
    duration = info["duration"]
    silent = info.get("silent")
    if not song:
        song = await get_stream_link(link)
    return song, title, link, thumb, from_user, play_this, duration, silent


# --------------------------------------------------


async def download(query):
    if query.startswith("https://") and not "youtu" in query.lower():
        thumb, duration = None, "Unknown"
        title = link = query
    else:
        obj = VideosSearch(query, limit=1)
        search = await obj.next()
        data = search["result"][0]
        link = data["link"]
        title = data["title"]
        duration = data.get("duration") or "♾"
        thumb = f"https://i.ytimg.com/vi/{data['id']}/hqdefault.jpg"
    dl = await get_stream_link(link)
    return dl, thumb, title, link, duration


async def get_stream_link(ytlink):
    """
    info = YoutubeDL({}).extract_info(url=ytlink, download=False)
    k = ""
    for x in info["formats"]:
        h, w = ([x["height"], x["width"]])
        if h and w:
            if h <= 720 and w <= 1280:
                k = x["url"]
    return k
    """
    stream = await bash(f'yt-dlp -g -f "best[height<=?480][width<=?854]" {ytlink}')
    return stream[0]


async def vid_download(query):
    obj = VideosSearch(query, limit=1)
    search = await obj.next()
    data = search["result"][0]
    link = data["link"]
    video = await get_stream_link(link)
    title = data["title"]
    thumb = f"https://i.ytimg.com/vi/{data['id']}/hqdefault.jpg"
    duration = data.get("duration") or "♾"
    return video, thumb, title, link, duration


async def dl_playlist(chat, from_user, link):
    # untill issue get fix
    # https://github.com/alexmercerind/youtube-search-python/issues/107
    """
    vids = Playlist.getVideos(link)
    try:
        vid1 = vids["videos"][0]
        duration = vid1["duration"] or "♾"
        title = vid1["title"]
        song = await get_stream_link(vid1['link'])
        thumb = f"https://i.ytimg.com/vi/{vid1['id']}/hqdefault.jpg"
        return song[0], thumb, title, vid1["link"], duration
    finally:
        vids = vids["videos"][1:]
        for z in vids:
            duration = z["duration"] or "♾"
            title = z["title"]
            thumb = f"https://i.ytimg.com/vi/{z['id']}/hqdefault.jpg"
            add_to_queue(chat, None, title, z["link"], thumb, from_user, duration)
    """
    links = await get_videos_link(link)
    try:
        obj = VideosSearch(links[0], limit=1)
        search = await obj.next()
        vid1 = search["result"][0]
        duration = vid1.get("duration") or "♾"
        title = vid1["title"]
        song = await get_stream_link(vid1["link"])
        thumb = f"https://i.ytimg.com/vi/{vid1['id']}/hqdefault.jpg"
        return song, thumb, title, vid1["link"], duration
    finally:
        for z in links[1:]:
            try:
                obj = VideosSearch(z, limit=1)
                search = await obj.next()
                vid = search["result"][0]
                duration = vid.get("duration") or "♾"
                title = vid["title"]
                thumb = f"https://i.ytimg.com/vi/{vid['id']}/hqdefault.jpg"
                add_to_queue(chat, None, title, vid["link"], thumb, from_user, duration)
            except Exception as er:
                LOGS.exception(er)


async def file_download(event, reply, fast_download=True):
    thumb = "https://telegra.ph/file/22bb2349da20c7524e4db.mp4"
    title = reply.file.title or reply.file.name or str(time()) + (reply.file.ext or "")
    dl_loc = check_filename(f"vcbot/downloads/{get_tg_filename(reply)}")
    dl, _ = await tg_downloader(
        media=reply,
        event=event,
        filename=dl_loc,
        show_progress=fast_download,
    )
    duration = (
        time_formatter(reply.file.duration * 1000) if reply.file.duration else "🤷‍♂️"
    )
    if reply.document.thumbs:
        thumb = await reply.download_media(str(Path(dl_loc).with_suffix(".jpg")), thumb=-1)
    return dl, thumb, title, reply.message_link, duration


# --------------------------------------------------

# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

import asyncio
import base64
import os
import re
import random
import string
from logging import WARNING
from traceback import format_exc
from urllib.parse import unquote

from telethon.tl import types
from telethon.utils import get_display_name, get_peer_id

from . import some_random_headers

from pyUltroid import *
from pyUltroid._misc._wrappers import eor
from pyUltroid.dB import DEVLIST
from pyUltroid.dB._core import LIST
from pyUltroid.exceptions import DependencyMissingError
from pyUltroid.custom.commons import (
    aiohttp,
    async_searcher,
    asyncread,
    asyncwrite,
    check_filename,
    json_parser,
    osremove,
)

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import catbox
except ImportError:
    catbox = None


# --------------------------------------------------


async def randomchannel(
    tochat, channel, range1, range2, caption=None, client=ultroid_bot
):
    do = random.randrange(range1, range2)
    async for x in client.iter_messages(channel, add_offset=do, limit=1):
        caption = caption or x.text
        try:
            await client.send_message(tochat, caption, file=x.media)
        except BaseException:
            pass


# --------------------------------------------------


async def YtDataScraper(url: str):
    to_return = {}
    data = json_parser(
        BeautifulSoup(
            await async_searcher(url),
            "html.parser",
        )
        .find_all("script")[41]
        .text[20:-1]
    )["contents"]
    _common_data = data["twoColumnWatchNextResults"]["results"]["results"]["contents"]
    common_data = _common_data[0]["videoPrimaryInfoRenderer"]
    try:
        description_data = _common_data[1]["videoSecondaryInfoRenderer"]["description"][
            "runs"
        ]
    except (KeyError, IndexError):
        description_data = [{"text": "U hurrr from here"}]
    description = "".join(
        description_datum["text"] for description_datum in description_data
    )
    to_return["title"] = common_data["title"]["runs"][0]["text"]
    to_return["views"] = (
        common_data["viewCount"]["videoViewCountRenderer"]["shortViewCount"][
            "simpleText"
        ]
        or common_data["viewCount"]["videoViewCountRenderer"]["viewCount"]["simpleText"]
    )
    to_return["publish_date"] = common_data["dateText"]["simpleText"]
    to_return["likes"] = (
        common_data["videoActions"]["menuRenderer"]["topLevelButtons"][0][
            "toggleButtonRenderer"
        ]["defaultText"]["simpleText"]
        # or like_dislike[0]["toggleButtonRenderer"]["defaultText"]["accessibility"][
        #    "accessibilityData"
        # ]["label"]
    )
    to_return["description"] = description
    return to_return


# --------------------------------------------------


async def google_search(query):
    query = query.replace(" ", "+")
    _base = "https://google.com"
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "User-Agent": random.choice(some_random_headers),
    }
    con = await async_searcher(_base + "/search?q=" + query, headers=headers)
    soup = BeautifulSoup(con, "html.parser")
    result = []
    pdata = soup.find_all("a", href=re.compile("url="))
    for data in pdata:
        if not data.find("div"):
            continue
        try:
            result.append(
                {
                    "title": data.find("div").text,
                    "link": unquote(data["href"].split("&url=")[1].split("&ved=")[0]),
                    "description": data.find_all("div")[-1].text,
                }
            )
        except IndexError:
            continue
        except Exception as er:
            LOGS.exception(er)
    return result


# ----------------------------------------------------


async def ReTrieveFile(input_file_name):
    if not aiohttp:
        raise DependencyMissingError("This function needs 'aiohttp' to be installed.")

    async def _func(out):
        contentType = out.headers.get("content-type")
        if "image" not in contentType:
            return (False, await out.json())
        content = await out.read()
        path = check_filename("ult-rmbg.png")
        await asyncwrite(path, content, mode="wb+")
        return (True, path)

    RMBG_API = udB.get_key("RMBG_API")
    headers = {"X-API-Key": RMBG_API}
    files = {"image_file": await asyncread(input_file_name, binary=True)}
    response = await async_searcher(
        "https://api.remove.bg/v1.0/removebg",
        post=True,
        evaluate=_func,
        headers=headers,
        data=files,
    )
    if not response[0]:
        LOGS.error(response[1])
    return response


# ---------------- Unsplash Search ----------------
# @New-Dev0


async def unsplashsearch(query, limit=None, shuf=True):
    query = query.replace(" ", "-")
    link = "https://unsplash.com/s/photos/" + query
    extra = await async_searcher(link, re_content=True)
    res = BeautifulSoup(extra, "html.parser")
    all_ = res.find_all("img", srcset=re.compile("images.unsplash.com/photo"))
    if shuf:
        random.shuffle(all_)
    return list(map(lambda e: e["src"], all_[:limit]))


# ---------------- Random User Gen ----------------


# @xditya
async def get_random_user_data():
    base_url = "https://randomuser.me/api/"
    cc = await async_searcher(
        "https://random-data-api.com/api/business_credit_card/random_card", re_json=True
    )
    card = (
        "**CARD_ID:** "
        + str(cc["credit_card_number"])
        + f" {cc['credit_card_expiry_date']}\n"
        + f"**C-ID :** {cc['id']}"
    )
    data_ = (await async_searcher(base_url, re_json=True))["results"][0]
    _g = data_["gender"]
    gender = "🤵🏻‍♂" if _g == "male" else "🤵🏻‍♀"
    name = data_["name"]
    loc = data_["location"]
    dob = data_["dob"]
    msg = """
{} **Name:** {}.{} {}
**Street:** {} {}
**City:** {}
**State:** {}
**Country:** {}
**Postal Code:** {}
**Email:** {}
**Phone:** {}
**Card:** {}
**Birthday:** {}
""".format(
        gender,
        name["title"],
        name["first"],
        name["last"],
        loc["street"]["number"],
        loc["street"]["name"],
        loc["city"],
        loc["state"],
        loc["country"],
        loc["postcode"],
        data_["email"],
        data_["phone"],
        card,
        dob["date"][:10],
    )
    pic = data_["picture"]["large"]
    return msg, pic


# Dictionary (Synonyms and Antonyms)
async def get_synonyms_or_antonyms(word, type_of_words):
    if type_of_words not in ["synonyms", "antonyms"]:
        return "Dude! Please give a corrent type of words you want."
    s = await async_searcher(
        f"https://tuna.thesaurus.com/pageData/{word}", re_json=True
    )
    li_1 = [
        y
        for x in [
            s["data"]["definitionData"]["definitions"][0][type_of_words],
            s["data"]["definitionData"]["definitions"][1][type_of_words],
        ]
        for y in x
    ]
    return [y["term"] for y in li_1]


# Quotly
class Quotly:
    _API = "https://bot.lyo.su/quote/generate"
    _entities = {
        types.MessageEntityPhone: "phone_number",
        types.MessageEntityMention: "mention",
        types.MessageEntityBold: "bold",
        types.MessageEntityCashtag: "cashtag",
        types.MessageEntityStrike: "strikethrough",
        types.MessageEntityHashtag: "hashtag",
        types.MessageEntityEmail: "email",
        types.MessageEntityMentionName: "text_mention",
        types.MessageEntityUnderline: "underline",
        types.MessageEntityUrl: "url",
        types.MessageEntityTextUrl: "text_link",
        types.MessageEntityBotCommand: "bot_command",
        types.MessageEntityCode: "code",
        types.MessageEntityPre: "pre",
        types.MessageEntitySpoiler: "spoiler",
    }

    async def _format_quote(self, event, reply=None, sender=None, type_="private"):
        async def telegraph(file_):
            file = file_ + ".png"
            Image.open(file_).save(file, "PNG")
            uri = catbox.helpers.upload_to_litterbox(
                file, file_name=os.path.basename(file.name)
            )
            osremove(file, file_)
            return uri

        if reply and reply.raw_text:
            reply = {
                "name": get_display_name(reply.sender) or "Deleted Account",
                "text": reply.raw_text,
                "chatId": reply.chat_id,
            }
        else:
            reply = {}
        is_fwd = event.fwd_from
        name = None
        last_name = None
        if sender and sender.id not in DEVLIST:
            id_ = get_peer_id(sender)
        elif not is_fwd:
            id_ = event.sender_id
            sender = await event.get_sender()
        else:
            id_, sender = None, None
            name = is_fwd.from_name
            if is_fwd.from_id:
                id_ = get_peer_id(is_fwd.from_id)
                try:
                    sender = await event.client.get_entity(id_)
                except ValueError:
                    pass
        if sender:
            name = get_display_name(sender)
            if hasattr(sender, "last_name"):
                last_name = sender.last_name
        entities = []
        if event.entities:
            for entity in event.entities:
                if type(entity) in self._entities:
                    enti_ = entity.to_dict()
                    del enti_["_"]
                    enti_["type"] = self._entities[type(entity)]
                    entities.append(enti_)
        text = event.raw_text
        if isinstance(event, types.MessageService):
            if isinstance(event.action, types.MessageActionGameScore):
                text = f"scored {event.action.score}"
                rep = await event.get_reply_message()
                if rep and rep.game:
                    text += f" in {rep.game.title}"
            elif isinstance(event.action, types.MessageActionPinMessage):
                text = "pinned a message."
            # TODO: Are there any more events with sender?
        message = {
            "entities": entities,
            "chatId": id_,
            "avatar": True,
            "from": {
                "id": id_,
                "first_name": (name or (sender.first_name if sender else None))
                or "Deleted Account",
                "last_name": last_name,
                "username": sender.username if sender else None,
                "language_code": "en",
                "title": name,
                "name": name or "Deleted Account",
                "type": type_,
            },
            "text": text,
            "replyMessage": reply,
        }
        if event.document and event.document.thumbs:
            file_ = await event.download_media(thumb=-1)
            uri = await telegraph(file_)
            message["media"] = {"url": uri}

        return message

    # old api - https://bot.lyo.su/quote/generate
    async def create_quotly(
        self,
        event,
        url="https://quoteampi.onrender.com/generate",
        reply={},
        bg=None,
        sender=None,
        file_name="quote.webp",
    ):
        """Create quotely's quote."""
        if not isinstance(event, list):
            event = [event]

        from pyUltroid import udB

        if udB.get_key("OQAPI"):
            url = Quotly._API

        if not bg:
            bg = "#1b1429"
        content = {
            "type": "quote",
            "format": "webp",
            "backgroundColor": bg,
            "width": 512,
            "height": 768,
            "scale": 2,
            "messages": [
                await self._format_quote(message, reply=reply, sender=sender)
                for message in event
            ],
        }

        request = await async_searcher(url, post=True, json=content, re_json=True)
        if request.get("ok"):
            image = base64.decodebytes(request["result"]["image"].encode("utf-8"))
            await asyncwrite(file_name, image, mode="wb+")
            return file_name
        raise Exception(str(request))


# https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotate_image(image, angle):
    if not cv2:
        raise DependencyMissingError("This function needs 'cv2' to be installed!")
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


__all__ = (
    "Quotly",
    "YtDataScraper",
    "google_search",
    "randomchannel",
    "rotate_image",
    "unsplashsearch",
)

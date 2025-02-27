# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

import json
import html
import math
import os
import re
import random
import secrets
import ssl
import subprocess
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from shlex import quote as shquote
from shutil import copy2
from traceback import format_exc
from urllib.parse import quote, unquote

from telethon import Button
from telethon.tl.types import DocumentAttributeAudio, DocumentAttributeVideo

from . import some_random_headers

from pyUltroid import *
from pyUltroid.custom.mediainfo import gen_mediainfo
from pyUltroid.dB.filestore_db import get_stored_msg, store_msg
from pyUltroid.custom.commons import (
    async_searcher,
    asyncwrite,
    bash,
    check_filename,
    humanbytes,
    json_parser,
    osremove,
    run_async,
)

try:
    import certifi
except ImportError:
    certifi = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image, ImageDraw, ImageFont = None, None, None
    LOGS.info("PIL not installed!")

try:
    import requests
    from requests.exceptions import MissingSchema
except ImportError:
    requests = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from telegraph import Telegraph
except ImportError:
    Telegraph = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


# ~~~~~~~~~~~~~~~~~~~~OFOX API~~~~~~~~~~~~~~~~~~~~
# @buddhhu


async def get_ofox(codename):
    ofox_baseurl = "https://api.orangefox.download/v3/"
    releases = await async_searcher(
        ofox_baseurl + "releases?codename=" + codename, re_json=True
    )
    device = await async_searcher(
        ofox_baseurl + "devices/get?codename=" + codename, re_json=True
    )
    return device, releases


# ~~~~~~~~~~~~~~~ Link Checker ~~~~~~~~~~~~~~~~


async def is_url_ok(url: str):
    try:
        return await async_searcher(url, method="head")
    except Exception as er:
        LOGS.debug(er)


# ~~~~~~~~~~~~~~~~ Metadata ~~~~~~~~~~~~~~~~~~~~


async def metadata(file):
    data = await gen_mediainfo(file)
    if type(data) == dict:
        if data["type"] in ("video", "gif"):
            if not data["bitrate"]:
                data["bitrate"] = 320
            if not data["width"]:
                data["width"] = 1280
            if not data["height"]:
                data["height"] = 720
        return data

    out, _ = await bash(f"mediainfo {shquote(file)} --Output=JSON")
    if _ and _.endswith("not found"):
        raise DependencyMissingError(
            "'mediainfo' is not installed!\nInstall it to use this command."
        )
    data = {}
    _info = json.loads(out)["media"]
    if not _info:
        return {}
    _info = _info["track"]
    info = _info[0]
    if info.get("Format") in ("GIF", "PNG"):
        return {
            "height": _info[1]["Height"],
            "width": _info[1]["Width"],
            "bitrate": _info[1].get("BitRate", 320),
        }
    if info.get("AudioCount"):
        data["title"] = info.get("Title", file)
        data["performer"] = info.get("Performer") or udB.get_key("artist") or ""
    if info.get("VideoCount"):
        data["height"] = int(float(_info[1].get("Height", 720)))
        data["width"] = int(float(_info[1].get("Width", 1280)))
        data["bitrate"] = int(_info[1].get("BitRate", 320))
    data["duration"] = int(float(info.get("Duration", 0)))
    return data


# ~~~~~~~~~~~~~~~~ Attributes ~~~~~~~~~~~~~~~~


async def set_attributes(file):
    data = await metadata(file)
    if not data:
        return None
    if "width" in data:
        return [
            DocumentAttributeVideo(
                duration=data.get("duration", 0),
                w=data.get("width", 512),
                h=data.get("height", 512),
                supports_streaming=True,
            )
        ]
    return [
        DocumentAttributeAudio(
            duration=data.get("duration", 0),
            title=data.get("title", Path(file).stem),
            performer=data.get("performer"),
        )
    ]


# ~~~~~~~~~~~~~~~~ Button stuffs ~~~~~~~~~~~~~~~


def get_msg_button(texts: str):
    btn = []
    for z in re.findall("\\[(.*?)\\|(.*?)\\]", texts):
        text, url = z
        urls = url.split("|")
        url = urls[0]
        if len(urls) > 1:
            btn[-1].append([text, url])
        else:
            btn.append([[text, url]])

    txt = texts
    for z in re.findall("\\[.+?\\|.+?\\]", texts):
        txt = txt.replace(z, "")

    return txt.strip(), btn


def create_tl_btn(button: list):
    btn = []
    for z in button:
        if len(z) > 1:
            kk = [Button.url(x, y.strip()) for x, y in z]
            btn.append(kk)
        else:
            btn.append([Button.url(z[0][0], z[0][1].strip())])
    return btn


def format_btn(buttons: list):
    txt = ""
    for i in buttons:
        a = 0
        for i in i:
            if hasattr(i.button, "url"):
                a += 1
                if a > 1:
                    txt += f"[{i.button.text} | {i.button.url} | same]"
                else:
                    txt += f"[{i.button.text} | {i.button.url}]"
    _, btn = get_msg_button(txt)
    return btn


# ~~~~~~~~~~~~~~~Saavn Downloader~~~~~~~~~~~~~~~
# @techierror


async def saavn_search(query: str):
    try:
        data = await async_searcher(
            url=f"https://saavn-api.vercel.app/search/{query.replace(' ', '%20')}",
            re_json=True,
        )
    except BaseException:
        data = None
    return data


# --- webupload ------#
# @buddhhu

_webupload_cache = {}


async def webuploader(chat_id: int, msg_id: int, uploader: str):
    if chat_id in _webupload_cache and msg_id in _webupload_cache[chat_id]:
        file = _webupload_cache[chat_id][msg_id]
    else:
        return "File not found in cache."

    sites = {
        "siasky": {"url": "https://siasky.net/skynet/skyfile", "json": True},
        "file.io": {"url": "https://file.io/", "json": True},
        "0x0.st": {"url": "https://0x0.st", "json": False},
        "transfer": {"url": "https://transfer.sh", "json": False},
        "catbox": {"url": "https://catbox.moe/user/api.php", "json": False},
        "filebin": {"url": "https://filebin.net", "json": False},
    }

    if uploader and uploader in sites:
        url = sites[uploader]["url"]
        json_format = sites[uploader]["json"]
    else:
        return "Uploader not supported or invalid."

    files = {"file": open(file, "rb")}  # Adjusted for both formats

    try:
        if uploader == "filebin":
            cmd = f"curl -X POST --data-binary '@{file}' -H 'filename: \"{file}\"' \"{url}\""
            response = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if response.returncode == 0:
                response_json = json.loads(response.stdout)
                bin_id = response_json.get("bin", {}).get("id")
                if bin_id:
                    filebin_url = f"https://filebin.net/{bin_id}"
                    return filebin_url
                else:
                    return "Failed to extract bin ID from Filebin response"
            else:
                return f"Failed to upload file to Filebin: {response.stderr.strip()}"
        elif uploader == "catbox":
            cmd = f"curl -F reqtype=fileupload -F time=24h -F 'fileToUpload=@{file}' {url}"
        elif uploader == "0x0.st":
            cmd = f"curl -F 'file=@{file}' {url}"
        elif uploader == "file.io" or uploader == "siasky":
            try:
                status = await async_searcher(
                    url, data=files, post=True, re_json=json_format
                )
            except Exception as e:
                return f"Failed to upload file: {e}"
            if isinstance(status, dict):
                if "skylink" in status:
                    return f"https://siasky.net/{status['skylink']}"
                if status.get("status") == 200:
                    return status.get("link")
        else:
            raise ValueError("Uploader not supported")

        response = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if response.returncode == 0:
            return response.stdout.strip()
        else:
            return f"Failed to upload file: {response.stderr.strip()}"
    except Exception as e:
        return f"Failed to upload file: {e}"

    del _webupload_cache.get(chat_id, {})[msg_id]
    return "Failed to get valid URL for the uploaded file."


def text_set(text):
    lines = []
    if len(text) <= 55:
        lines.append(text)
    else:
        all_lines = text.split("\n")
        for line in all_lines:
            if len(line) <= 55:
                lines.append(line)
            else:
                k = len(line) // 55
                for z in range(1, k + 2):
                    lines.append(line[((z - 1) * 55) : (z * 55)])
    return lines[:25]


# ------------------Logo Gen Helpers----------------
# @TechiError


class LogoHelper:
    @staticmethod
    def get_text_size(text, image, font):
        im = Image.new("RGB", (image.width, image.height))
        draw = ImageDraw.Draw(im)
        dim = draw.textbbox((0, 0), text, font=font)
        return (dim[2] - dim[0], dim[3] - dim[1])

    @staticmethod
    def find_font_size(text, font, image, target_width_ratio):
        tested_font_size = 100
        tested_font = ImageFont.truetype(font, tested_font_size)
        observed_width, observed_height = LogoHelper.get_text_size(
            text, image, tested_font
        )
        estimated_font_size = (
            tested_font_size / (observed_width / image.width) * target_width_ratio
        )
        return round(estimated_font_size)

    @staticmethod
    def make_logo(imgpath, text, funt, **args):
        fill = args.get("fill")
        width_ratio = args.get("width_ratio") or 0.7
        stroke_width = int(args.get("stroke_width"))
        stroke_fill = args.get("stroke_fill")

        img = Image.open(imgpath)
        width, height = img.size
        fct = min(height, width)
        if height != width:
            img = img.crop((0, 0, fct, fct))
        if img.height < 1000:
            img = img.resize((1020, 1020))
        width, height = img.size
        draw = ImageDraw.Draw(img)
        font_size = LogoHelper.find_font_size(text, funt, img, width_ratio)
        font = ImageFont.truetype(funt, font_size)
        dim = draw.textbbox((0, 0), text, font=font)
        l, t, r, b = font.getbbox(text)
        w, h = r - l, (b - t) * 1.5
        draw.text(
            ((width - w) / 2, ((height - h) / 2)),
            text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        file_name = check_filename("logo.png")
        img.save(file_name, "PNG", optimize=True)
        return file_name


# --------------------------------------
# @New-Dev0


async def get_paste(data: str, extension: str = "txt"):
    url = "https://spaceb.in/api/"
    try:
        res = await async_searcher(
            url,
            json={"content": data, "extension": extension},
            post=True,
            re_json=True,
        )
        key = res["payload"]["id"]
        _ext = "" if extension == "txt" else f".{extension}"
        return True, {
            "link": f"https://spaceb.in/{key}{_ext}",
            "raw": f"https://spaceb.in/{key}/raw",
        }
    except Exception:
        try:
            url = "https://dpaste.org/api/"
            data = {
                "format": "json",
                "content": data.encode("utf-8"),
                "lexer": extension,
                "expires": "604800",  # expire in week
            }
            res = await async_searcher(url, data=data, post=True, re_json=True)
            return True, {"link": res["url"], "raw": f"{res['url']}/raw"}
        except Exception as e:
            LOGS.exception(e)
            return None, {"link": None, "raw": None, "error": str(e)}


# --------------------------------------


# https://stackoverflow.com/a/74563494
async def get_google_images(query):
    """Get image results from Google Custom Search API.

    Args:
        query (str): Search query string

    Returns:
        list: List of dicts containing image info (title, link, source, thumbnail, original)
    """
    # Google Custom Search API credentials
    google_keys = [
        {"key": "AIzaSyAj75v6vHWLJdJaYcj44tLz7bdsrh2g7Y0", "cx": "712a54749d99a449e"},
        {"key": "AIzaSyDFQQwPLCzcJ9FDao-B7zDusBxk8GoZ0HY", "cx": "001bbd139705f44a6"},
        {"key": "AIzaSyD0sRNZUa8-0kq9LAREDAFKLNO1HPmikRU", "cx": "4717c609c54e24250"},
    ]

    key_index = random.randint(0, len(google_keys) - 1)
    GOOGLE_API_KEY = google_keys[key_index]["key"]
    GOOGLE_CX = google_keys[key_index]["cx"]
    try:
        # Construct API URL
        url = (
            "https://www.googleapis.com/customsearch/v1"
            f"?q={quote(query)}"
            f"&cx={GOOGLE_CX}"
            f"&key={GOOGLE_API_KEY}"
            "&searchType=image"
            "&num=10"  # Number of results
        )

        # Make API request
        response = await async_searcher(url, re_json=True)
        if not response or "items" not in response:
            LOGS.error(
                "No results from Google Custom Search API for {query}: {response}"
            )
            return []

        # Process results
        google_images = []
        for item in response["items"]:
            try:
                google_images.append(
                    {
                        "title": item.get("title", ""),
                        "link": item.get("contextLink", ""),  # Page containing image
                        "source": item.get("displayLink", ""),
                        "thumbnail": item.get("image", {}).get(
                            "thumbnailLink", item["link"]
                        ),
                        "original": item["link"],  # Original image URL
                    }
                )
            except Exception:
                continue

        # Randomize results order
        random.shuffle(google_images)
        return google_images
    except Exception as e:
        LOGS.exception(f"Error in get_google_images: {str(e)}")
        return []


# Thanks https://t.me/ImSafone for ChatBotApi
async def get_chatbot_reply(message):
    req_link = f"https://api.safone.dev/chatbot?query={message}"
    try:
        return (await async_searcher(req_link, re_json=True)).get("response")
    except Exception:
        LOGS.info(f"chatbot error: {format_exc()}")


# ------ Audio \\ Video tools func --------#


# https://github.com/1Danish-00/CompressorBot/blob/main/helper/funcn.py#L104


async def genss(file):
    n = await metadata(file)
    return n.get("duration")


async def duration_s(file, stime):
    tsec = await genss(file)
    rnd = random.choice((0.33, 0.5, 0.6, 0.75, 0.25, 0.45))
    x = round(tsec * rnd)
    y = round(tsec * rnd + int(stime))
    pin = _stdr(x)
    pon = _stdr(y) if y < tsec else _stdr(tsec)
    return pin, pon


def _stdr(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if len(str(minutes)) == 1:
        minutes = "0" + str(minutes)
    if len(str(hours)) == 1:
        hours = "0" + str(hours)
    if len(str(seconds)) == 1:
        seconds = "0" + str(seconds)
    return (
        ((str(hours) + ":") if hours else "00:")
        + ((str(minutes) + ":") if minutes else "00:")
        + ((str(seconds)) if seconds else "")
    )


# ------------------- used in pdftools --------------------#


# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example


def order_points(pts):
    "get the required points"
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    try:
        import cv2
    except ImportError:
        raise DependencyMissingError("This function needs 'cv2' to be installed.")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


# ------------------------------------- Telegraph ---------------------------------- #


class _TelegraphClient:
    __slots__ = ("client",)

    def __init__(self):
        self.client = None

    def get_client(self):
        if not Telegraph:
            return LOGS.error("'Telegraph' is not Installed!")

        if self.client:
            return self.client

        from pyUltroid import udB, ultroid_bot

        token = udB.get_key("_TELEGRAPH_TOKEN")
        domain = udB.get_key("GRAPH_DOMAIN") or "graph.org"
        client = Telegraph(token, domain=domain)
        self.client = client
        if token:
            return

        gd_name = ultroid_bot.full_name
        short_name = gd_name[:30]
        profile_url = (
            f"https://t.me/{ultroid_bot.me.username}"
            if ultroid_bot.me.username
            else "https://t.me/TeamUltroid"
        )
        try:
            client.create_account(
                short_name=short_name, author_name=gd_name, author_url=profile_url
            )
        except Exception as er:
            if "SHORT_NAME_TOO_LONG" in str(er):
                client.create_account(
                    short_name="ultroiduser",
                    author_name=gd_name,
                    author_url=profile_url,
                )
            else:
                return LOGS.exception(er)
        finally:
            udB.set_key("_TELEGRAPH_TOKEN", client.get_access_token())

    @run_async
    def create_page(self, title, **kwargs):
        self.get_client()
        page = self.client.create_page(title=title, **kwargs)
        return page["url"]

    @run_async
    def upload_file(self, file, anon=False):
        if anon == True:
            client = Telegraph(domain="graph.org")
        else:
            self.get_client()
            client = self.client
        page = client.upload_file(file)
        return [i["src"] for i in page]


TelegraphClient = _TelegraphClient()


async def Carbon(
    code,
    base_url="https://carbonara.solopov.dev/api/cook",
    file_name="ultroid",
    download=False,
    rayso=False,
    **kwargs,
):
    kwargs["code"] = code
    if rayso:
        base_url = "https://api.safone.dev/rayso"
        kwargs["theme"] = kwargs.get("theme", "breeze")
        kwargs["darkMode"] = kwargs.get("darkMode", True)
        kwargs["title"] = kwargs.get("title", "Ultroid")
        con = await async_searcher(base_url, post=True, json=kwargs, re_json=True)
        if con and "image" in con:
            con = b64decode(con["image"])
    else:
        con = await async_searcher(base_url, post=True, json=kwargs, re_content=True)

    if type(con) == dict:
        return con
    if not download:
        file = BytesIO(con)
        file.name = file_name + ".jpg"
    else:
        try:
            return json_parser(con.decode())
        except Exception:
            pass
        file = check_filename(file_name + ".jpg")
        await asyncwrite(file, con, mode="wb+")
    return file


async def get_file_link(msg):
    from pyUltroid import udB

    msg_id = await msg.forward_to(udB.get_key("LOG_CHANNEL"))
    await msg_id.reply(
        "**Message has been stored to generate a shareable link. Do not delete it.**"
    )
    msg_id = msg_id.id
    msg_hash = secrets.token_hex(nbytes=8)
    store_msg(msg_hash, msg_id)
    return msg_hash


async def get_stored_file(event, hash):
    from pyUltroid import asst, udB

    msg_id = get_stored_msg(hash)
    if not msg_id:
        return
    try:
        msg = await asst.get_messages(udB.get_key("LOG_CHANNEL"), ids=msg_id)
    except Exception as er:
        LOGS.warning(f"FileStore, Error: {er}")
        return
    if not msg_id:
        return await asst.send_message(
            event.chat_id, "__Message was deleted by owner!__", reply_to=event.id
        )
    await asst.send_message(event.chat_id, msg.text, file=msg.media, reply_to=event.id)


def translate(text, lang_tgt="en", lang_src="auto", timeout=60, detect=False):
    pattern = r'(?s)class="(?:t0|result-container)">(.*?)<'
    escaped_text = quote(text.encode("utf8"))
    url = "https://translate.google.com/m?tl=%s&sl=%s&q=%s" % (
        lang_tgt,
        lang_src,
        escaped_text,
    )
    response = requests.get(url, timeout=timeout).content
    result = response.decode("utf8")
    result = re.findall(pattern, result)
    if not result:
        return ""
    text = html.unescape(result[0])
    return (text, None) if detect else text


def cmd_regex_replace(cmd):
    out = (
        cmd.replace("$", "")
        .replace("?(.*)", "")
        .replace("( (.*)|)", "")
        .replace("(.*)", "")
        .replace("(?: |)", "")
        .replace(r"\d*|", "")
        .replace(r"[\s\S]*", "")
        .replace("( |)", "")
        .replace("?((.|//)*)", "")
        .replace("?P<shortname>\\w+", "")
        .replace("(", "")
        .replace(")", "")
        .replace("?(\\d+)", "")
        .replace(r"\d*", "")
    )
    for i in "|$?":
        out = out.removesuffix(i)
    return out.strip()


# ------------------------#


class LottieException(Exception):
    pass


class TgConverter:
    """Convert files related to Telegram"""

    @staticmethod
    async def animated_sticker(file, out_path="sticker.tgs", throw=False, remove=False):
        """Convert to/from animated sticker."""
        LOGS.info(f"Converting animated sticker: {file} -> {out_path}")
        try:
            if out_path.endswith("webp"):
                er, out = await bash(
                    f"lottie_convert.py --webp-quality 100 --webp-skip-frames 100 {shquote(file)} {shquote(out_path)}"
                )
            else:
                er, out = await bash(
                    f"lottie_convert.py {shquote(file)} {shquote(out_path)}"
                )

            if er:
                LOGS.error(f"Error in animated_sticker conversion: {er}")
                if throw:
                    raise LottieException(er)
            if remove and os.path.exists(file):
                os.remove(file)
                LOGS.info(f"Removed original file: {file}")
            if os.path.exists(out_path):
                LOGS.info(f"Successfully converted to {out_path}")
                return out_path
            LOGS.error(f"Output file not created: {out_path}")
            return None
        except Exception as e:
            LOGS.exception(f"Unexpected error in animated_sticker: {str(e)}")
            if throw:
                raise

    @staticmethod
    async def animated_to_gif(file, out_path="gif.gif"):
        """Convert animated sticker to gif."""
        LOGS.info(f"Converting to gif: {file} -> {out_path}")
        try:
            er, out = await bash(
                f"lottie_convert.py {shquote(file)} {shquote(out_path)}"
            )
            if er:
                LOGS.error(f"Error in animated_to_gif conversion: {er}")
            if os.path.exists(out_path):
                LOGS.info("Successfully converted to gif")
                return out_path
            LOGS.error("Gif conversion failed - output file not created")
            return None
        except Exception as e:
            LOGS.exception(f"Unexpected error in animated_to_gif: {str(e)}")
            return None

    @staticmethod
    def resize_photo_sticker(photo):
        """Resize the given photo to 512x512 (for creating telegram sticker)."""
        LOGS.info(f"Resizing photo for sticker: {photo}")
        try:
            image = Image.open(photo)
            original_size = (image.width, image.height)

            if (image.width and image.height) < 512:
                size1 = image.width
                size2 = image.height
                if image.width > image.height:
                    scale = 512 / size1
                    size1new = 512
                    size2new = size2 * scale
                else:
                    scale = 512 / size2
                    size1new = size1 * scale
                    size2new = 512
                size1new = math.floor(size1new)
                size2new = math.floor(size2new)
                sizenew = (size1new, size2new)
                image = image.resize(sizenew)
            else:
                maxsize = (512, 512)
                image.thumbnail(maxsize)

            LOGS.info(f"Resized image from {original_size} to {image.size}")
            return image
        except Exception as e:
            LOGS.exception(f"Error in resize_photo_sticker: {str(e)}")
            raise

    @staticmethod
    async def ffmpeg_convert(input_, output, remove=False):
        if output.endswith(".webm"):
            return await TgConverter.create_webm(
                input_, name=output[:-5], remove=remove
            )
        if output.endswith(".gif"):
            out, er = await bash(
                f"ffmpeg -i {shquote(input_)} -an -sn -c:v copy {shquote(output)}.mp4 -y"
            )
            LOGS.info(f"FFmpeg output: {out}, Error: {er}")
        else:
            out, er = await bash(f"ffmpeg -i {shquote(input_)} {shquote(output)} -y")
            LOGS.info(f"FFmpeg output: {out}, Error: {er}")
        if remove:
            os.remove(input_)
        if os.path.exists(output):
            return output

    @staticmethod
    async def create_webm(file, name="video", remove=False):
        LOGS.info(f"Creating webm: {file} -> {name}.webm")
        try:
            _ = await metadata(file)
            name += ".webm"
            h, w = _["height"], _["width"]

            if h == w and h != 512:
                h, w = 512, 512
            if h != 512 or w != 512:
                if h > w:
                    h, w = 512, -1
                if w > h:
                    h, w = -1, 512

            await bash(
                f'ffmpeg -i {shquote(file)} -preset fast -an -to 00:00:03 -crf 30 -bufsize 256k -b:v {_["bitrate"]} -vf "scale={w}:{h},fps=30" -c:v libvpx-vp9 {shquote(name)} -y'
            )

            if remove and os.path.exists(file):
                os.remove(file)
                LOGS.info(f"Removed original file: {file}")

            if os.path.exists(name):
                LOGS.info(f"Successfully created webm: {name}")
                return name

            LOGS.error(f"Webm creation failed - output file not created: {name}")
            return None
        except Exception as e:
            LOGS.exception(f"Error in create_webm: {str(e)}")
            return None

    @staticmethod
    def to_image(input_, name, remove=False):
        """Convert video/gif to image using first frame."""
        LOGS.info(f"Converting to image: {input_} -> {name}")
        try:
            if not input_:
                LOGS.error("Input file is None")
                return None

            if not os.path.exists(input_):
                LOGS.error(f"Input file does not exist: {input_}")
                return None

            try:
                import cv2
            except ImportError:
                raise DependencyMissingError(
                    "This function needs 'cv2' to be installed."
                )

            img = cv2.VideoCapture(input_)
            success, frame = img.read()

            if not success:
                LOGS.error(f"Failed to read frame from {input_}")
                return None

            cv2.imwrite(name, frame)
            img.release()

            if not os.path.exists(name):
                LOGS.error(f"Failed to save image: {name}")
                return None

            if remove and os.path.exists(input_):
                os.remove(input_)
                LOGS.info(f"Removed original file: {input_}")

            LOGS.info(f"Successfully converted to image: {name}")
            return name

        except Exception as e:
            LOGS.exception(f"Error in to_image conversion: {str(e)}")
            return None

    @staticmethod
    async def convert(
        input_file,
        outname="converted",
        convert_to=None,
        allowed_formats=[],
        remove_old=True,
    ):
        """Convert between different file formats."""
        LOGS.info(f"Converting {input_file} to {convert_to or allowed_formats}")

        if not input_file:
            LOGS.error("Input file is None")
            return None

        if not os.path.exists(input_file):
            LOGS.error(f"Input file does not exist: {input_file}")
            return None

        if "." in input_file:
            ext = input_file.split(".")[-1].lower()
        else:
            LOGS.error("Input file has no extension")
            return input_file

        if (
            ext in allowed_formats
            or ext == convert_to
            or not (convert_to or allowed_formats)
        ):
            return input_file

        def recycle_type(exte):
            return convert_to == exte or exte in allowed_formats

        try:
            # Sticker to Something
            if ext == "tgs":
                for extn in ("webp", "json", "png", "mp4", "gif"):
                    if recycle_type(extn):
                        name = outname + "." + extn
                        result = await TgConverter.animated_sticker(
                            input_file, name, remove=remove_old
                        )
                        if result:
                            return result
                if recycle_type("webm"):
                    gif_file = await TgConverter.convert(
                        input_file, convert_to="gif", remove_old=remove_old
                    )
                    if gif_file:
                        return await TgConverter.create_webm(
                            gif_file, outname, remove=True
                        )

            # Json -> Tgs
            elif ext == "json":
                if recycle_type("tgs"):
                    name = outname + ".tgs"
                    return await TgConverter.animated_sticker(
                        input_file, name, remove=remove_old
                    )

            # Video to Something
            elif ext in ("webm", "mp4", "gif"):
                for exte in ("webm", "mp4", "gif"):
                    if recycle_type(exte):
                        name = outname + "." + exte
                        result = await TgConverter.ffmpeg_convert(
                            input_file, name, remove=remove_old
                        )
                        if result:
                            return result

                for exte in ("png", "jpg", "jpeg", "webp"):
                    if recycle_type(exte):
                        name = outname + "." + exte
                        result = TgConverter.to_image(
                            input_file, name, remove=remove_old
                        )
                        if result:
                            return result

            # Image to Something
            elif ext in ("jpg", "jpeg", "png", "webp"):
                for extn in ("png", "webp", "ico"):
                    if recycle_type(extn):
                        try:
                            img = Image.open(input_file)
                            name = outname + "." + extn
                            img.save(name, extn.upper())
                            if remove_old and os.path.exists(input_file):
                                os.remove(input_file)
                                LOGS.info(f"Removed original file: {input_file}")
                            return name
                        except Exception as e:
                            LOGS.error(f"Failed to convert image to {extn}: {str(e)}")
                            continue

                for extn in ("webm", "gif", "mp4"):
                    if recycle_type(extn):
                        name = outname + "." + extn
                        if extn == "webm":
                            png_file = await TgConverter.convert(
                                input_file,
                                convert_to="png",
                                remove_old=remove_old,
                            )
                            if png_file:
                                return await TgConverter.ffmpeg_convert(
                                    png_file, name, remove=True
                                )
                        else:
                            return await TgConverter.ffmpeg_convert(
                                input_file, name, remove=remove_old
                            )

            LOGS.error(
                f"No valid conversion found for {input_file} to {convert_to or allowed_formats}"
            )
            return None
        except Exception as e:
            LOGS.exception(f"Error in convert: {str(e)}")
            return None


def safe_load(file, *args, **kwargs):
    def _get_value(stri):
        try:
            value = eval(stri.strip())
        except Exception:
            # from .. import LOGS

            # LOGS.debug(er)
            value = stri.strip()
        return value

    if isinstance(file, str):
        read = file.split("\n")
    else:
        read = file.readlines()
    out = {}
    for line in read:
        if ":" in line:  # Ignores Empty & Invalid lines
            spli = line.split(":", maxsplit=1)
            key = spli[0].strip()
            value = _get_value(spli[1])
            out.update({key: value or []})
        elif "-" in line:
            spli = line.split("-", maxsplit=1)
            where = out[list(out.keys())[-1]]
            if isinstance(where, list):
                value = _get_value(spli[1])
                if value:
                    where.append(value)
    return out


def get_chat_and_msgid(link):
    matches = re.findall("https:\\/\\/t\\.me\\/(c\\/|)(.*)\\/(.*)", link)
    if not matches:
        return None, None
    _, chat, msg_id = matches[0]
    if chat.isdigit():
        chat = int("-100" + chat)
    return chat, int(msg_id)


# --------- END --------- #

__all__ = (
    "Carbon",
    "LogoHelper",
    "TgConverter",
    "TelegraphClient",
    "cmd_regex_replace",
    "create_tl_btn",
    "format_btn",
    "genss",
    "get_chat_and_msgid",
    "get_chatbot_reply",
    "get_file_link",
    "get_msg_button",
    "get_ofox",
    "get_paste",
    "get_stored_file",
    "is_url_ok",
    # "make_html_telegraph",
    "metadata",
    "saavn_search",
    "safe_load",
    "set_attributes",
    "text_set",
    "translate",
    "webuploader",
)

# some custom functions that do not belong anywhere else..

import asyncio
from ast import literal_eval
from functools import wraps
from io import BytesIO
import mimetypes
from pathlib import Path
from random import choice, randrange
from shlex import split as shsplit
import string
from time import perf_counter
from urllib.parse import urlencode

from telethon.tl import types
from telethon.tl.functions.messages import SaveGifRequest
from telethon.utils import get_display_name, get_input_document
from telethon.tl.types import InputMessagesFilterPhotos

from pyUltroid import LOGS, asst, udB, ultroid_bot
from ._loop import loop, run_async_task, tasks_db
from .commons import (
    async_searcher,
    asyncread,
    asyncwrite,
    json_parser,
    not_so_fast,
    osremove,
    random_string,
    run_async,
    string_is_url,
)

try:
    import requests
except ImportError:
    requests = None


# scheduler
def init_scheduler():
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except ImportError:
        return None

    schd = AsyncIOScheduler(timezone="Asia/Kolkata")
    schd.start()
    return schd


scheduler = init_scheduler()


# https://gist.github.com/DougAF/ef88f89d1d99763bb05afd81285ef233#file-timer-py
def timeit(func):
    """To Check running time of functions."""

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def exec_time(*args, **kwargs):
            start = perf_counter()
            result = await func(*args, **kwargs)
            time_taken = perf_counter() - start
            return f"Function: {func.__name__} \nOutput: {result} \nTime taken: {time_taken:.5f} seconds."

        return exec_time
    else:

        @wraps(func)
        def exec_time(*args, **kwargs):
            start = perf_counter()
            result = func(*args, **kwargs)
            time_taken = perf_counter() - start
            return f"Function: {func.__name__} \nOutput: {result} \nTime taken: {time_taken:.5f} seconds."

        return exec_time


async def cleargif(gif):
    if not gif.client._bot and gif.gif:
        try:
            await gif.client(SaveGifRequest(id=get_input_document(gif), unsave=True))
        except Exception as exc:
            return LOGS.warning(f"error in cleargif: {exc}", exc_info=True)


async def get_imgbb_link(path, url=None, **kwargs):
    api = udB.get_key("IMGBB_API")
    if not api:
        raise Exception(f"'IMGBB_API' is missing, Can't upload this image to imgbb.")
    if not (path or url):
        raise TypeError("I Need image path or url to upload..")
    if path and not Path(path).is_file():
        raise FileNotFoundError(f"could not found: {path!r}")

    if path:
        image_data = await asyncread(path, binary=True)
        if kwargs.get("delete"):
            osremove(path)
    else:
        image_data = url
    try:
        post = await async_searcher(
            "https://api.imgbb.com/1/upload",
            post=True,
            data={
                "key": api,
                "image": image_data,
                "name": kwargs.get("title", random_string(length=9)),
                "expiration": str(kwargs.get("expire", 0)),
            },
            re_json=True,
        )
    except Exception as exc:
        return LOGS.exception("Error in posting data to ImgBB Server!")

    if post.get("status") != 200:
        return LOGS.error(json_parser(post, indent=4))
    flink = post["data"]["url"] if kwargs.get("hq") else post["data"]["display_url"]
    if kwargs.get("preview"):
        try:
            await not_so_fast(
                asst.send_message, udB.get_key("TAG_LOG"), flink, link_preview=True
            )
            await asyncio.sleep(3)
        except Exception as exc:
            LOGS.warning("ImgBB preview error:", exc_info=True)
    return flink


@run_async
def Catbox(
    file,
    temp=False,
    time=None,
    file_name=None,
    mime_type=None,
):
    """
    Upload file to Catbox (permanent) or Litterbox (temporary)

    file: Path to the file or BytesIO object or URL of a file.
    temp: Whether file should be Uploaded Permanently (catbox vs litterbox).

    * Optional kwargs *
    file_name: Name of the file with extension (required for URL upload).
    mime_type: Mime type of file (required for URL upload)
    time: Duration for which the file will be available. (only for temp mode, options - '1h', '12h', '24h', '72h', '1w').

    returns URL of the uploaded file.
    """

    if not requests:
        raise Exception(f"'requests' package is missing, can't upload to Catbox!")

    api = (
        "https://litterbox.catbox.moe/resources/internals/api.php"
        if temp
        else "https://catbox.moe/user/api.php"
    )

    try:
        path = Path(file)
    except TypeError:
        path = None

    data = {"reqtype": "fileupload"}
    if path and path.is_file():
        with open(file, "rb") as f:
            content = f.read()
        mime_type = mimetypes.guess_type(file)[0]
        files = {"fileToUpload": (path.name, content, mime_type)}
    elif isinstance(file, BytesIO):
        if not (file_name and mime_type):
            raise ValueError(
                "Provide 'file_name' and 'mime_type' in arguments for this to work."
            )

        mime_type = mime_type or "application/octet-stream"
        files = {"fileToUpload": (file_name, file, mime_type)}
    elif string_is_url(file):
        data.update({"reqtype": "urlupload", "url": file})
    else:
        raise TypeError(
            f"File type Unsupported: {type(file)}, need a URL or filepath to upload to Catbox!"
        )

    if temp:
        data["time"] = time or "1h"

    if api_key := udB.get_key("CATBOX_API"):
        data["userhash"] = api_key

    response = requests.post(api, files=files, data=data, timeout=60)
    return response.text.strip()


RaySoThemes = (
    "meadow",
    "breeze",
    "raindrop",
    "candy",
    "crimson",
    "falcon",
    "sunset",
    "midnight",
    "mono",
    "raindrop",
    "noir",
    "bitmap",
    "ice",
    "sand",
    "forest",
)


async def generate_rayso(
    text,
    title="Ultroid",
    dark_mode=True,
    theme=None,
    filename=None,
):
    from playwright.async_api import async_playwright
    from playwright._impl._errors import TimeoutError

    if not filename:
        filename = random_string(12) + ".png"
    if not theme or theme not in RaySoThemes:
        theme = choice(RaySoThemes)

    data = {
        "darkMode": dark_mode,
        "theme": theme,
        "title": title,
    }
    url = f"https://ray.so/#{urlencode(data)}"
    async with async_playwright() as play:
        chrome = await play.chromium.launch()
        page = await chrome.new_page()
        try:
            await page.goto(url)
            await page.wait_for_load_state("networkidle")

            try:
                await page.wait_for_selector(
                    "div[class*='Editor_editor__']", timeout=60000
                )
                editor = await page.query_selector("div[class*='Editor_editor__']")
                await editor.focus()
                await editor.click()

                for line in text.splitlines():
                    await page.keyboard.type(line.strip())
                    await page.keyboard.press("Enter")

                await page.evaluate(
                    """() => {
                    const button = document.querySelector('button[aria-label="Export as PNG"]');
                    button.click();
                }"""
                )

                async with page.expect_download() as dl:
                    dled = await dl.value
                    await dled.save_as(filename)
            except TimeoutError as exc:
                return LOGS.exception(
                    f"Timeout error while genersting rayso (playwright!): {exc}"
                )
        except Exception as exc:
            return LOGS.exception(f"Error while generating Rayso (playwright): {exc}")

    return filename


class RandomPhotoHandler:
    __slots__ = ("ok", "running", "photos_to_store", "sources")

    def __init__(self):
        self.ok = bool(udB.get_key("__RANDOM_PIC", force=True))
        self.running = False
        self.photos_to_store = 15
        self.sources = (
            ("r_wallpapers", 9547, 31000),
            ("Anime_hot_wallpapers", 5, 12000),
        )

    async def get(self, clear=True):
        photos = udB.get_key("__RANDOM_PIC", force=True) or []
        if not photos:
            run_async_task(self._save_images, id="random_pic")
            return udB.get_key("ALIVE_PIC")
        pic = choice(photos)
        if clear:
            photos.remove(pic)
            udB.set_key("__RANDOM_PIC", photos)
            if not self.running:
                run_async_task(self._save_images, id="random_pic")
        return pic

    async def _save_images(self):
        self.running = True
        pics = udB.get_key("__RANDOM_PIC", force=True)
        if len(pics) >= self.photos_to_store:
            return

        for _ in range(self.photos_to_store - len(pics)):
            chat, min_id, max_id = choice(self.sources)
            async for msg in ultroid_bot.iter_messages(
                chat,
                limit=1,
                filter=InputMessagesFilterPhotos,
                offset_id=randrange(min_id, max_id),
            ):
                pic_name = chat + "_" + str(msg.id)
                await asyncio.sleep(randrange(10, 35))
                path = await msg.download_media()
                imgbb_link = await get_imgbb_link(
                    path,
                    expire=10 * 86400,
                    title=pic_name,
                    delete=True,
                    preview=True,
                )
                if imgbb_link:
                    pics.append(imgbb_link)
        udB.set_key("__RANDOM_PIC", pics)
        self.running = False


random_pic = RandomPhotoHandler()


class unix_parser:
    __slots__ = ("args", "kwargs")

    def __init__(self, text):
        self.args = []
        self.kwargs = {}
        split = shsplit(text)
        for word in split:
            if word.startswith("-") and len(word) > 1:
                split = word.split("=", maxsplit=1)
                key, value = split if len(split) > 1 else (word, "True")
                self.kwargs[key[1:]] = self._checker(value)
            else:
                self.args.append(word)

        self.args = " ".join(self.args)

    @staticmethod
    def _checker(data):
        try:
            return literal_eval(data)
        except Exception:
            return data


__all__ = (
    "Catbox",
    "timeit",
    "cleargif",
    "osremove",
    "get_imgbb_link",
    "generate_rayso",
    "loop",
    "random_pic",
    "run_async_task",
    "scheduler",
    "tasks_db",
    "unix_parser",
)

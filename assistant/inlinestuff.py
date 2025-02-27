# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

import base64
import inspect
from datetime import datetime
from html import unescape
from random import choice
from re import compile as re_compile

from bs4 import BeautifulSoup as bs
from telethon import Button
from telethon.tl.alltlobjects import LAYER, tlobjects
from telethon.tl.types import DocumentAttributeAudio as Audio
from telethon.tl.types import InputWebDocument as wb

from pyUltroid.custom._extras import FixedSizeDict
from pyUltroid.fns.misc import google_search
from pyUltroid.fns.tools import (
    _webupload_cache,
    async_searcher,
    get_ofox,
    saavn_search,
    webuploader,
)

from . import *
from . import _ult_cache


MOD_APIS = (
    "QUl6YVN5QXlEQnNZM1dSdEI1WVBDNmFCX3c4SkF5NlpkWE5jNkZV",
    "QUl6YVN5QkYwenhMbFlsUE1wOXh3TVFxVktDUVJxOERnZHJMWHNn",
    "QUl6YVN5RGRPS253blB3VklRX2xiSDVzWUU0Rm9YakFLSVFWMERR",
)


CACHE = {
    "app": FixedSizeDict(maxsize=16),
    "fdroid": FixedSizeDict(maxsize=16),
    "saavn": FixedSizeDict(),  # default = 32
    "google": FixedSizeDict(maxsize=20),
    "mods": FixedSizeDict(maxsize=10),
}


@in_pattern("ofox", owner=True)
async def orenge_fomx(e):
    ofox = "https://graph.org/file/231f0049fcd722824f13b.jpg"
    try:
        match = e.text.split(" ", maxsplit=1)[1]
    except IndexError:
        kkkk = await e.builder.article(
            title="Enter Device Codename",
            thumb=wb(ofox, 0, "image/jpeg", []),
            text="**OFᴏx🦊Rᴇᴄᴏᴠᴇʀʏ**\n\nYou didn't search anything",
            buttons=Button.switch_inline("Sᴇᴀʀᴄʜ Aɢᴀɪɴ", query="ofox ", same_peer=True),
        )
        return await e.answer([kkkk])
    device, releases = await get_ofox(match)
    if device.get("detail") is None:
        fox = []
        fullname = device["full_name"]
        codename = device["codename"]
        # str(device["supported"])
        maintainer = device["maintainer"]["name"]
        link = f"https://orangefox.download/device/{codename}"
        for data in releases["data"]:
            release = data["type"]
            version = data["version"]
            size = humanbytes(data["size"])
            release_date = datetime.utcfromtimestamp(data["date"]).strftime("%Y-%m-%d")
            text = f"[\xad]({ofox})**OʀᴀɴɢᴇFᴏx Rᴇᴄᴏᴠᴇʀʏ Fᴏʀ**\n\n"
            text += f"`  Fᴜʟʟ Nᴀᴍᴇ: {fullname}`\n"
            text += f"`  Cᴏᴅᴇɴᴀᴍᴇ: {codename}`\n"
            text += f"`  Mᴀɪɴᴛᴀɪɴᴇʀ: {maintainer}`\n"
            text += f"`  Bᴜɪʟᴅ Tʏᴘᴇ: {release}`\n"
            text += f"`  Vᴇʀsɪᴏɴ: {version}`\n"
            text += f"`  Sɪᴢᴇ: {size}`\n"
            text += f"`  Bᴜɪʟᴅ Dᴀᴛᴇ: {release_date}`"
            fox.append(
                await e.builder.article(
                    title=f"{fullname}",
                    description=f"{version}\n{release_date}",
                    text=text,
                    thumb=wb(ofox, 0, "image/jpeg", []),
                    link_preview=True,
                    buttons=[
                        Button.url("Dᴏᴡɴʟᴏᴀᴅ", url=f"{link}"),
                        Button.switch_inline(
                            "Sᴇᴀʀᴄʜ Aɢᴀɪɴ", query="ofox ", same_peer=True
                        ),
                    ],
                )
            )
        await e.answer(
            fox,
            switch_pm="OrangeFox Recovery Search.",
            switch_pm_param="start",
            cache_time=150,
        )
    else:
        await e.answer(
            [], switch_pm="OrangeFox Recovery Search.", switch_pm_param="start"
        )


@in_pattern("fl2lnk ?(.*)", fullsudo=True)
async def fille_leenks(e):
    match = e.pattern_match.group(1)
    chat_id, msg_id = match.split(":")
    filename = _webupload_cache[int(chat_id)][int(msg_id)]
    if "/" in filename:
        filename = filename.split("/")[-1]
    __cache = f"{chat_id}:{msg_id}"
    buttons = [
        [
            Button.inline("catbox", data=f"flcatbox//{__cache}"),
            Button.inline("transfer", data=f"fltransfer//{__cache}"),
        ],
        [
            Button.inline("filebin", data=f"flfilebin//{__cache}"),
            Button.inline("0x0.st", data=f"fl0x0.st//{__cache}"),
        ],
        [
            Button.inline("file.io", data=f"flfile.io//{__cache}"),
            Button.inline("siasky", data=f"flsiasky//{__cache}"),
        ],
    ]
    try:
        lnk = [
            await e.builder.article(
                title=f"Upload {filename}",
                text=f"**File:**\n{filename}",
                buttons=buttons,
            )
        ]
    except BaseException as er:
        LOGS.exception(er)
        lnk = [
            await e.builder.article(
                title="fl2lnk",
                text="File not found",
            )
        ]
    await e.answer(lnk, switch_pm="File to Link.", switch_pm_param="start")


@callback(
    re_compile(
        "fl(.*)",
    ),
    fullsudo=True,
)
async def file_linkk(e):
    t = (e.data).decode("UTF-8")
    data = t[2:]
    host = data.split("//")[0]
    chat_id, msg_id = data.split("//")[1].split(":")
    filename = _webupload_cache[int(chat_id)][int(msg_id)]
    if "/" in filename:
        filename = filename.split("/")[-1]
    await e.edit(f"Uploading `{filename}` on {host}")
    link = (await webuploader(chat_id, msg_id, host)).strip().replace("\n", "")
    await e.edit(f"Uploaded `{filename}` on {host}.", buttons=Button.url("View", link))


@in_pattern("repo", owner=True)
async def rempu(e):
    ultpic = "https://graph.org/file/4136aa1650bc9d4109cc5.jpg"
    SUP_BUTTONS = [
        [
            Button.url("• Repo •", url="https://github.com/TeamUltroid/Ultroid"),
            Button.url("• Support •", url="t.me/UltroidSupportChat"),
        ],
    ]
    res = [
        await e.builder.article(
            title="Ultroid Userbot",
            description="Userbot | Telethon",
            thumb=wb(ultpic, 0, "image/jpeg", []),
            text="• **ULTROID USERBOT** •",
            buttons=SUP_BUTTONS,
        ),
    ]
    await e.answer(res, switch_pm="Ultroid Repo.", switch_pm_param="start")


@in_pattern("go", owner=True)
async def gsearch(q_event):
    try:
        match = q_event.text.split(maxsplit=1)[1]
    except IndexError:
        return await q_event.answer(
            [], switch_pm="Google Search. Enter a query!", switch_pm_param="start"
        )

    if match in CACHE["google"]:
        return await q_event.answer(
            CACHE["google"][match], switch_pm="Google Search.", switch_pm_param="start"
        )
    searcher = []
    gugirl = "https://graph.org/file/0df54ae4541abca96aa11.jpg"
    gresults = await google_search(match)
    for i in gresults[:50]:
        try:
            title = i["title"]
            link = i["link"]
            desc = i["description"]
            searcher.append(
                await q_event.builder.article(
                    title=title or match,
                    description=desc,
                    thumb=wb(gugirl, 0, "image/jpeg", []),
                    text=f"**Gᴏᴏɢʟᴇ Sᴇᴀʀᴄʜ**\n\n**••Tɪᴛʟᴇ••**\n`{title}`\n\n**••Dᴇsᴄʀɪᴘᴛɪᴏɴ••**\n`{desc}`",
                    link_preview=False,
                    buttons=[
                        [Button.url("Lɪɴᴋ", url=f"{link}")],
                        [
                            Button.switch_inline(
                                "Sᴇᴀʀᴄʜ Aɢᴀɪɴ",
                                query="go ",
                                same_peer=True,
                            ),
                            Button.switch_inline(
                                "Sʜᴀʀᴇ",
                                query=f"go {match}",
                                same_peer=False,
                            ),
                        ],
                    ],
                ),
            )
        except IndexError:
            break
    await q_event.answer(searcher, switch_pm="Google Search.", switch_pm_param="start")
    CACHE["google"][match] = searcher


@in_pattern("mods", fullsudo=True)
async def search_mod(e):
    try:
        quer = e.text.split(" ", maxsplit=1)[1]
    except IndexError:
        return await e.answer(
            [], switch_pm="Mod Apps Search. Enter app name!", switch_pm_param="start"
        )

    if quer in CACHE["mods"]:
        return await e.answer(
            CACHE["mods"][quer],
            switch_pm="Search Mod Applications.",
            switch_pm_param="start",
        )

    start = 0 * 3 + 1
    da = base64.b64decode(choice(MOD_APIS)).decode("ascii")
    url = f"https://www.googleapis.com/customsearch/v1?key={da}&cx=25b3b50edb928435b&q={quer}&start={start}"
    data = await async_searcher(url, re_json=True)
    search_items = data.get("items", [])
    modss = []
    for a in search_items:
        title = a.get("title")
        desc = a.get("snippet")
        link = a.get("link")
        text = f"**••Tɪᴛʟᴇ••** `{title}`\n\n"
        text += f"**Dᴇsᴄʀɪᴘᴛɪᴏɴ** `{desc}`"
        modss.append(
            await e.builder.article(
                title=title,
                description=desc,
                text=text,
                link_preview=True,
                buttons=[
                    [Button.url("Dᴏᴡɴʟᴏᴀᴅ", url=f"{link}")],
                    [
                        Button.switch_inline(
                            "Mᴏʀᴇ Mᴏᴅs",
                            query="mods ",
                            same_peer=True,
                        ),
                        Button.switch_inline(
                            "Sʜᴀʀᴇ",
                            query=f"mods {quer}",
                            same_peer=False,
                        ),
                    ],
                ],
            ),
        )
    await e.answer(
        modss[:50], switch_pm="Search Mod Applications.", switch_pm_param="start"
    )
    CACHE["mods"][quer] = modss[:50]


PLAY_API = "https://googleplay.onrender.com/api/apps?q="


@in_pattern("app", owner=True)
async def find_apps(e):
    try:
        f = e.text.split(maxsplit=1)[1].lower()
    except IndexError:
        return await e.answer(
            [],
            switch_pm="Enter app name to Search on Play Store!",
            switch_pm_param="start",
        )

    if f in CACHE["app"]:
        return await e.answer(
            CACHE["app"][f], switch_pm="Application Searcher.", switch_pm_param="start"
        )

    foles = []
    url = PLAY_API + f.replace(" ", "+")
    aap = await async_searcher(url, re_json=True)
    for z in aap["results"][:50]:
        url = "https://play.google.com/store/apps/details?id=" + z["appId"]
        name = z["title"]
        desc = unescape(z["summary"])[:300].replace("<br>", "\n") + "..."
        dev = z["developer"]["devId"]
        text = f"**••Aᴘᴘ Nᴀᴍᴇ••** [{name}]({url})\n"
        text += f"**••Dᴇᴠᴇʟᴏᴘᴇʀ••** `{dev}`\n"
        text += f"**••Dᴇsᴄʀɪᴘᴛɪᴏɴ••**\n`{desc}`"
        foles.append(
            await e.builder.article(
                title=name,
                description=dev,
                thumb=wb(z["icon"], 0, "image/jpeg", []),
                text=text,
                link_preview=True,
                buttons=[
                    [Button.url("Lɪɴᴋ", url=url)],
                    [
                        Button.switch_inline(
                            "Mᴏʀᴇ Aᴘᴘs",
                            query="app ",
                            same_peer=True,
                        ),
                        Button.switch_inline(
                            "Sʜᴀʀᴇ",
                            query=f"app {f}",
                            same_peer=False,
                        ),
                    ],
                ],
            ),
        )
    await e.answer(foles, switch_pm="Application Searcher.", switch_pm_param="start")
    CACHE["app"][f] = foles


PISTON_URI = "https://emkc.org/api/v2/piston/"
PISTON_LANGS = {}


@in_pattern("run", owner=True)
async def piston_run(event):
    global PISTON_LANGS
    try:
        _, lang, code = event.text.split(maxsplit=2)
    except IndexError:
        result = await event.builder.article(
            title="Bad Query",
            description="Usage: [Language] [code]",
            thumb=wb(
                "https://graph.org/file/e33c57fc5f1044547e4d8.jpg", 0, "image/jpeg", []
            ),
            text=f'**Inline Usage**\n\n`@{asst.me.username} run python print("hello world")`\n\n[Language List](https://graph.org/Ultroid-09-01-6)',
        )
        return await event.answer([result])

    if not PISTON_LANGS:
        resp = await async_searcher(f"{PISTON_URI}runtimes", re_json=True)
        PISTON_LANGS = {lang.pop("language"): lang for lang in resp}

    if lang in PISTON_LANGS.keys():
        version = PISTON_LANGS[lang]["version"]
    else:
        result = await event.builder.article(
            title="Unsupported Language",
            description="Usage: [Language] [code]",
            thumb=wb(
                "https://graph.org/file/e33c57fc5f1044547e4d8.jpg", 0, "image/jpeg", []
            ),
            text=f'**Inline Usage**\n\n`@{asst.me.username} run python print("hello world")`\n\n[Language List](https://graph.org/Ultroid-09-01-6)',
        )
        return await event.answer([result])
    output = await async_searcher(
        f"{PISTON_URI}execute",
        post=True,
        json={
            "language": lang,
            "version": version,
            "files": [{"content": code}],
        },
        re_json=True,
    )

    output = output["run"]["output"] or get_string("instu_4")
    if len(output) > 3000:
        output = f"{output[:3000]}..."
    result = await event.builder.article(
        title="Result",
        description=output,
        text=f"• **Language:**\n`{lang}`\n\n• **Code:**\n`{code}`\n\n• **Result:**\n`{output}`",
        thumb=wb(
            "https://graph.org/file/871ee4a481f58117dccc4.jpg", 0, "image/jpeg", []
        ),
        buttons=Button.switch_inline("Fork", query=event.text, same_peer=True),
    )
    await event.answer([result], switch_pm="• Piston •", switch_pm_param="start")


@in_pattern("fdroid", owner=True)
async def do_magic(event):
    try:
        match = event.text.split(" ", maxsplit=1)[1].lower()
    except IndexError:
        return await event.answer(
            [], switch_pm="Enter Query to Search", switch_pm_param="start"
        )

    if match in CACHE["fdroid"]:
        return await event.answer(
            CACHE["fdroid"][match],
            switch_pm=f"• Results for {match}",
            switch_pm_param="start",
        )
    link = "https://search.f-droid.org/?q=" + match.replace(" ", "+")
    content = await async_searcher(link, re_content=True)
    BSC = bs(content, "html.parser", from_encoding="utf-8")
    ress = []
    for dat in BSC.find_all("a", "package-header")[:10]:
        image = dat.find("img", "package-icon")["src"]
        if image.endswith("/"):
            image = "https://graph.org/file/a8dd4a92c5a53a89d0eff.jpg"
        title = dat.find("h4", "package-name").text.strip()
        desc = dat.find("span", "package-summary").text.strip()
        text = f"• **Name :** `{title}`\n\n"
        text += f"• **Description :** `{desc}`\n"
        text += f"• **License :** `{dat.find('span', 'package-license').text.strip()}`"
        imga = wb(image, 0, "image/jpeg", [])
        ress.append(
            await event.builder.article(
                title=title,
                type="photo",
                description=desc,
                text=text,
                content=imga,
                thumb=imga,
                include_media=True,
                buttons=[
                    Button.inline(
                        "• Download •", "fd" + dat["href"].split("packages/")[-1]
                    ),
                    Button.switch_inline("• Share •", query=event.text),
                ],
            )
        )
    msg = f"Showing {len(ress)} Results!" if ress else "No Results Found"
    await event.answer(ress[:50], switch_pm=msg, switch_pm_param="start")
    CACHE["fdroid"][match] = ress[:50]


"""
# not free anymore

# Thanks to OpenSource
_bearer_collected = [
    "AAAAAAAAAAAAAAAAAAAAALIKKgEAAAAA1DRuS%2BI7ZRKiagD6KHYmreaXomo%3DP5Vaje4UTtEkODg0fX7nCh5laSrchhtLxeyEqxXpv0w9ZKspLD",
    "AAAAAAAAAAAAAAAAAAAAAL5iUAEAAAAAmo6FYRjqdKlI3cNziIm%2BHUQB9Xs%3DS31pj0mxARMTOk2g9dvQ1yP9wknvY4FPBPUlE00smJcncw4dPR",
    "AAAAAAAAAAAAAAAAAAAAAN6sVgEAAAAAMMjMMWrwgGyv7YQOWN%2FSAsO5SGM%3Dg8MG9Jq93Rlllaok6eht7HvRCruN4Vpzp4NaVsZaaHHWSTzKI8",
]


@in_pattern("twitter", fullsudo=True)
async def twitter_search(event):
    try:
        match = event.text.split(maxsplit=1)[1].lower()
    except IndexError:
        return await event.answer(
            [], switch_pm="Enter Query to Search", switch_pm_param="start"
        )
    try:
        return await event.answer(
            _ult_cache["twitter"][match],
            switch_pm="• Twitter Search •",
            switch_pm_param="start",
        )
    except KeyError:
        pass
    headers = {"Authorization": f"bearer {choice(_bearer_collected)}"}
    res = await async_searcher(
        f"https://api.twitter.com/1.1/users/search.json?q={match}",
        headers=headers,
        re_json=True,
    )
    reso = []
    for user in res:
        thumb = wb(user["profile_image_url_https"], 0, "image/jpeg", [])
        if user.get("profile_banner_url"):
            url = user["profile_banner_url"]
            text = f"[\xad]({url})• **Name :** `{user['name']}`\n"
        else:
            text = f"• **Name :** `{user['name']}`\n"
        text += f"• **Description :** `{user['description']}`\n"
        text += f"• **Username :** `@{user['screen_name']}`\n"
        text += f"• **Followers :** `{user['followers_count']}`    • **Following :** `{user['friends_count']}`\n"
        pro_ = "https://twitter.com/" + user["screen_name"]
        text += f"• **Link :** [Click Here]({pro_})\n_"
        reso.append(
            await event.builder.article(
                title=user["name"],
                description=user["description"],
                url=pro_,
                text=text,
                thumb=thumb,
            )
        )
    swi_ = f"🐦 Showing {len(reso)} Results!" if reso else "No User Found :("
    await event.answer(reso, switch_pm=swi_, switch_pm_param="start")
    if _ult_cache.get("twitter"):
        _ult_cache["twitter"].update({match: reso})
    else:
        _ult_cache.update({"twitter": {match: reso}})
"""


@in_pattern("saavn", owner=True)
async def saavn_inline_searcher(event):
    try:
        query = event.text.split(maxsplit=1)[1].lower()
    except IndexError:
        return await event.answer(
            [], switch_pm="Enter Query to search 🔍", switch_pm_param="start"
        )

    if query in CACHE["saavn"]:
        return await event.answer(
            CACHE["saavn"][query],
            switch_pm=f"Showing Results for {query}",
            switch_pm_param="start",
        )

    results = await saavn_search(query)
    swi = "🎵 Saavn Search" if results else "No Results Found!"
    res = []
    for song in results:
        thumb = wb(song["image"], 0, "image/jpeg", [])
        text = f"• **Title :** {song['title']}"
        text += f"\n• **Year :** {song['year']}"
        text += f"\n• **Lang :** {song['language']}"
        text += f"\n• **Artist :** {song['artists']}"
        text += f"\n• **Release Date :** {song['release_date']}"
        res.append(
            await event.builder.article(
                title=song["title"],
                description=song["artists"],
                type="audio",
                text=text,
                include_media=True,
                buttons=Button.switch_inline(
                    "Search Again 🔍", query="saavn", same_peer=True
                ),
                thumb=thumb,
                content=wb(
                    song["url"],
                    0,
                    "audio/mp4",
                    [
                        Audio(
                            title=song["title"],
                            duration=int(song["duration"]),
                            performer=song["artists"],
                        )
                    ],
                ),
            )
        )
    await event.answer(res[:50], switch_pm=swi, switch_pm_param="start")
    CACHE["saavn"][query] = res[:50]


@in_pattern("tl", owner=True)
async def inline_tl(ult):
    try:
        match = ult.text.split(maxsplit=1)[1]
    except IndexError:
        text = f"**Telegram TlObjects Searcher.**\n__(Don't use if you don't know what it is!)__\n\n• Example Usage\n`@{asst.me.username} tl GetFullUserRequest`"
        return await ult.answer(
            [
                await ult.builder.article(
                    title="How to Use?",
                    description="Tl Searcher by Ultroid",
                    url="https://t.me/TeamUltroid",
                    text=text,
                )
            ],
            switch_pm="Tl Search 🔍",
            switch_pm_param="start",
        )

    items = []
    for key in tlobjects.values():
        if match.lower() in key.__name__.lower():
            tyyp = "Function" if "tl.functions." in str(key) else "Type"
            text = f"**Name:** `{key.__name__}`\n"
            text += f"**Category:** `{tyyp}`\n"
            text += f"\n`from {key.__module__} import {key.__name__}`\n\n"
            if args := str(inspect.signature(key))[1:][:-1]:
                text += "**Parameter:**\n"
                for para in args.split(","):
                    text += " " * 4 + "`" + para + "`\n"
            text += f"\n**Layer:** `{LAYER}`"
            items.append((key.__name__, tyyp, text[:4080]))

    ac_total = len(items)
    offset = int(ult.query.offset or "0")
    if offset > len(items):
        items = []
    else:
        items = [
            await ult.builder.article(
                title=item[0],
                description=item[1],
                url="https://t.me/TeamUltroid",
                text=item[2],
                buttons=Button.url(
                    "TL Source!",
                    f"https://tl.telethon.dev/?q={item[0]}",
                ),
            )
            for item in items[offset : offset + 50]
        ]
    mo = f"Found {ac_total} results for {match}!"
    await ult.answer(
        items,
        switch_pm=mo,
        switch_pm_param="start",
        cache_time=10,
        next_offset=str(offset + 50),
    )


InlinePlugin.update(
    {
        "Pʟᴀʏ Sᴛᴏʀᴇ Aᴘᴘs": "app telegram",
        "Mᴏᴅᴅᴇᴅ Aᴘᴘs": "mods minecraft",
        "Sᴇᴀʀᴄʜ Oɴ Gᴏᴏɢʟᴇ": "go TeamUltroid",
        "WʜɪSᴘᴇʀ": "wspr @username Hello🎉",
        "YᴏᴜTᴜʙᴇ Dᴏᴡɴʟᴏᴀᴅᴇʀ": "yt Ed Sheeran Perfect",
        "Piston Eval": "run javascript console.log('Hello Ultroid')",
        "OʀᴀɴɢᴇFᴏx🦊": "ofox beryllium",
        # "Tᴡɪᴛᴛᴇʀ Usᴇʀ": "twitter theultroid",
        "Fᴅʀᴏɪᴅ Sᴇᴀʀᴄʜ": "fdroid telegram",
        "Sᴀᴀᴠɴ sᴇᴀʀᴄʜ": "saavn",
        "Tʟ Sᴇᴀʀᴄʜ": "tl",
    }
)

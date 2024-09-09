# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

from . import get_help

__doc__ = get_help("help_bot")

import os
import sys
import asyncio
import time
from platform import python_version as pyver
from random import choice

from telethon import __version__
from telethon.tl.functions import PingRequest
from telethon.utils import resolve_bot_file_id
from telethon.errors.rpcerrorlist import (
    BotMethodInvalidError,
    ChatSendMediaForbiddenError,
)

from pyUltroid.version import __version__ as UltVer
from . import (
    ATRA_COL,
    HOSTED_ON,
    LOGS,
    LOG_CHANNEL,
    OWNER_NAME,
    ULTROID_IMAGES,
    ULTConfig,
    Button,
    Carbon,
    TelegraphClient,
    Var,
    asyncread,
    asst,
    asyncwrite,
    bash,
    call_back,
    callback,
    custom_updater,
    eor,
    get_string,
    heroku_logs,
    in_pattern,
    inline_pic,
    osremove,
    random_pic,
    restart,
    shutdown,
    start_time,
    time_formatter,
    udB,
    ultroid_bot,
    ultroid_cmd,
    ultroid_version,
)


def ULTPIC():
    return inline_pic() or choice(ULTROID_IMAGES)


buttons = [
    [
        Button.url(get_string("bot_3"), "https://github.com/TeamUltroid/Ultroid"),
        Button.url(get_string("bot_4"), "t.me/UltroidSupportChat"),
    ]
]

# Will move to strings
alive_txt = """
The Ultroid Userbot

  ◍ Version - {}
  ◍ Py-Ultroid - {}
  ◍ Telethon - {}
"""


async def get_repo():
    repo, _ = await bash("git config --get remote.origin.url")
    branch, _ = await bash("git rev-parse --abbrev-ref HEAD")
    branch = branch.strip()
    repo = repo.replace(".git", f"/tree/{branch}")
    return repo, branch


@callback("alive")
async def alive(event):
    text = alive_txt.format(ultroid_version, UltVer, __version__)
    await event.answer(text, alert=True)


@ultroid_cmd(
    pattern="alive( (.*)|$)",
)
async def lol(ult):
    match = ult.pattern_match.group(1).strip()
    inline = None
    if match in ("inline", "i"):
        try:
            res = await ult.client.inline_query(asst.me.username, "alive")
            return await res[0].click(ult.chat_id)
        except BotMethodInvalidError:
            pass
        except BaseException as er:
            LOGS.exception(er)
        inline = True
    pic = await random_pic.get() if random_pic.ok else udB.get_key("ALIVE_PIC")
    if isinstance(pic, list):
        pic = choice(pic)
    uptime = time_formatter((time.time() - start_time) * 1000)
    header = udB.get_key("ALIVE_TEXT") or get_string("bot_1")
    repo, branch = await get_repo()
    kk = f" `[{branch}]({repo})` "
    if inline:
        kk = f"<a href={repo}>{branch}</a>"
        parse = "html"
        in_alive = "{}\n\n🌀 <b>Ultroid Version -><b> <code>{}</code>\n🌀 <b>PyUltroid -></b> <code>{}</code>\n🌀 <b>Python -></b> <code>{}</code>\n🌀 <b>Uptime -></b> <code>{}</code>\n🌀 <b>Branch -></b>[ {} ]\n\n• <b>Join @TeamUltroid</b>"
        als = in_alive.format(
            header,
            f"{ultroid_version} [{HOSTED_ON}]",
            UltVer,
            pyver(),
            uptime,
            kk,
        )

        if _e := udB.get_key("ALIVE_EMOJI"):
            als = als.replace("🌀", _e)
    else:
        parse = "md"
        als = (get_string("alive_1")).format(
            header,
            OWNER_NAME,
            f"{ultroid_version} [{HOSTED_ON}]",
            UltVer,
            uptime,
            pyver(),
            __version__,
            kk,
        )

        if a := udB.get_key("ALIVE_EMOJI"):
            als = als.replace("✵", a)
    if pic:
        try:
            await ult.reply(
                als,
                file=pic,
                parse_mode=parse,
                link_preview=False,
                buttons=buttons if inline else None,
            )
            return await ult.try_delete()
        except ChatSendMediaForbiddenError:
            pass
        except Exception as er:
            LOGS.exception(er)
            try:
                await ult.reply(file=pic)
                await ult.reply(
                    als,
                    parse_mode=parse,
                    buttons=buttons if inline else None,
                    link_preview=False,
                )
                return await ult.try_delete()
            except Exception as er:
                LOGS.exception(er)
    await eor(
        ult,
        als,
        parse_mode=parse,
        link_preview=False,
        buttons=buttons if inline else None,
    )


@ultroid_cmd(pattern="ping$", chats=[], type=["official", "assistant"])
async def _(event):
    x = await event.eor("Pong !")
    start = time.time()
    await event.client(PingRequest(ping_id=0))
    end = round((time.time() - start) * 1000, 3)
    uptime = time_formatter((time.time() - start_time) * 1000)
    await x.edit(get_string("ping").format(end, uptime))


@ultroid_cmd(
    pattern="cmds$",
)
async def cmds(event):
    txt = ""
    for z in LIST.keys():
        txt += f"PLUGIN NAME: {z}\n"
        for zz in LIST[z]:
            txt += HNDLR + zz + "\n"
        txt += "\n\n"
    url = await TelegraphClient.create_page(title="Ultroid All Cmds", content=[txt])
    await event.eor(f"All Ultroid Cmds : [Click Here]({url})", link_preview=False)


async def _updater(
    event,
    to_edit,
    to_update=True,
    task=None,
):
    await asyncio.sleep(3)
    if call_back:
        call_back()
    if to_update:
        out, err = await bash("git reset --hard ; git pull --rebase")
        # await bash("cd addons && git reset --hard; git pull --rebase")
        LOGS.info(f"Restarting: {out} | {err}")
    if task:
        await task

    who = "bot" if event.client._bot else "user"
    udB.set_key("_RESTART", f"{who}_{event.chat_id}_{event.id}")
    return await restart(ult=event, edit=to_edit)


@ultroid_cmd(
    pattern="restart( update)?$",
    fullsudo=True,
)
async def restartbt(ult):
    ok = await ult.eor(get_string("bot_5"))
    await _updater(
        ok, to_edit=True, task=asyncio.sleep(6), to_update=ult.pattern_match.group(1)
    )

    """
    await bash("git pull && pip3 install -r requirements.txt")
    if len(sys.argv) > 1:
        os.execl(sys.executable, sys.executable, "main.py")
    else:
        os.execl(sys.executable, sys.executable, "-m", "pyUltroid")
    """


@ultroid_cmd(
    pattern="shutdown$",
    fullsudo=True,
)
async def shutdownbot(ult):
    await shutdown(ult)


@ultroid_cmd(
    pattern="logs( (.*)|$)",
    chats=[],
)
async def send_logs(event):
    opt = event.pattern_match.group(1).strip()
    # file = f"ultroid{sys.argv[-1]}.log" if len(sys.argv) > 1 else "ultroid.log"
    file = "ultlogs.txt"
    if opt == "heroku":
        await heroku_logs(event)
    elif opt == "carbon" and Carbon:
        event = await event.eor(get_string("com_1"))
        code = (await asyncread(file))[-2300:]
        file = await Carbon(
            file_name="ultroid-logs",
            code=code,
            backgroundColor=choice(ATRA_COL),
        )
        await event.reply("**Ultroid Logs.**", file=file)
    elif opt in ("open", "o"):
        long = -800 if opt == "o" else -4000
        file = (await asyncread(file))[long:]
        return await event.eor(f"`{file}`")
    else:
        await event.respond(
            "**Ultroid Logs!**",
            file=file,
            thumb=ULTConfig.thumb,
        )
    await event.try_delete()


@in_pattern("alive", owner=True)
async def inline_alive(ult):
    pic = udB.get_key("ALIVE_PIC")
    if not pic and random_pic.ok:
        pic = await random_pic.get()
    if isinstance(pic, list):
        pic = choice(pic)
    uptime = time_formatter((time.time() - start_time) * 1000)
    header = udB.get_key("ALIVE_TEXT") or get_string("bot_1")
    repo, branch = await get_repo()
    kk = f"<a href={repo}>{branch}</a>"
    in_alive = "{}\n\n🌀 <b>Ultroid Version -><b> <code>{}</code>\n🌀 <b>PyUltroid -></b> <code>{}</code>\n🌀 <b>Python -></b> <code>{}</code>\n🌀 <b>Uptime -></b> <code>{}</code>\n🌀 <b>Branch -></b>[ {} ]\n\n• <b>Join @TeamUltroid</b>"
    als = in_alive.format(
        header,
        f"{ultroid_version} [{HOSTED_ON}]",
        UltVer,
        pyver(),
        uptime,
        kk,
    )

    if _e := udB.get_key("ALIVE_EMOJI"):
        als = als.replace("🌀", _e)
    builder = ult.builder
    if pic:
        try:
            if any(i in pic.lower() for i in (".jpg", ".jpeg", ".png")):
                results = [
                    await builder.photo(
                        pic, text=als, parse_mode="html", buttons=buttons
                    )
                ]
            else:
                if _pic := resolve_bot_file_id(pic):
                    pic = _pic
                    buttons.insert(
                        0, [Button.inline(get_string("bot_2"), data="alive")]
                    )
                results = [
                    await builder.document(
                        pic,
                        title="Inline Alive",
                        description="@TeamUltroid",
                        parse_mode="html",
                        buttons=buttons,
                    )
                ]
            return await ult.answer(results)
        except BaseException as er:
            LOGS.info(er)
    result = [
        await builder.article(
            "Alive", text=als, parse_mode="html", link_preview=False, buttons=buttons
        )
    ]
    await ult.answer(result)


@ultroid_cmd(pattern="update( (.*)|$)", fullsudo=True)
async def ult_updater(e):
    xx = await e.eor(get_string("upd_1"))
    args = e.pattern_match.group(2)

    if args and args in ("fast", "soft", "now"):
        await asyncio.sleep(3)
        await xx.edit(get_string("upd_7"))
        return await _updater(xx, to_edit=False, task=asyncio.sleep(6))

        """
        if HOSTED_ON != "heroku":
            await bash("git pull -f && pip3 install --no-cache-dir -r requirements.txt")
            os.execl(sys.executable, "python3", "-m", "pyUltroid")
            return
        else:
            return await restart(xx, EDIT=False)
        """

    out, html_out = await custom_updater()
    if not out:
        repo, branch = await get_repo()
        return await xx.edit(
            f"<code>Your BOT is </code><b>up-to-date</b><code> with </code><b><a href='{repo}'>[{branch}]</a></b>",
            parse_mode="html",
            link_preview=False,
        )

    if len(out) > 3800:
        file = "changelogs.txt"
        await asyncwrite(file, out, mode="w+")
        caption = "• **New Update Available!** •"
        await e.client.respond(caption, file=file, reply_to=e.reply_to_msg_id)
        osremove(file)
        return await xx.delete()
    await xx.edit(html_out, parse_mode="html")


@callback("updtavail", fullsudo=True)
async def update_available(event):
    await event.answer("Processing Updates, Please Wait..")
    await asyncio.sleep(2)
    pic = await random_pic.get() if random_pic.ok else ULTPIC()
    await asst.send_file(
        udB.get_key("LOG_CHANNEL"),
        pic,
        caption="• **Update Available** •",
        force_document=False,
        buttons=[Button.inline("Changelogs", data="doupdate")],
    )

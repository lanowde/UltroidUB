# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

import asyncio
import os
import random
import shutil
import time
from pathlib import Path
from random import randint

from telethon.errors import (
    ChannelsTooMuchError,
    ChatAdminRequiredError,
    MessageIdInvalidError,
    MessageNotModifiedError,
    UserNotParticipantError,
)
from telethon.tl.custom import Button
from telethon.tl.functions.channels import (
    CreateChannelRequest,
    EditAdminRequest,
    EditPhotoRequest,
    InviteToChannelRequest,
)
from telethon.tl.functions.contacts import UnblockRequest
from telethon.tl.types import (
    ChatAdminRights,
    ChatPhotoEmpty,
    InputChatUploadedPhoto,
    InputMessagesFilterDocument,
)
from telethon.utils import get_peer_id
from decouple import config, RepositoryEnv

from pyUltroid import LOGS, Var, ULTConfig
from pyUltroid.fns.helper import download_file, inline_mention, custom_updater
# from pyUltroid.custom.commons import bash


_db_url = 0


async def autoupdate_local_database():
    from .. import asst, udB, ultroid_bot, Var

    global _db_url
    _db_url = (
        udB.get_key("TGDB_URL") or Var.TGDB_URL or ultroid_bot._cache.get("TGDB_URL")
    )
    if _db_url:
        _split = _db_url.split("/")
        _channel = _split[-2]
        _id = _split[-1]
        try:
            await asst.edit_message(
                int(_channel) if _channel.isdigit() else _channel,
                message=_id,
                file="database.json",
                text="**Do not delete this file.**",
            )
        except MessageNotModifiedError:
            return
        except MessageIdInvalidError:
            pass
    try:
        LOG_CHANNEL = (
            udB.get_key("LOG_CHANNEL")
            or Var.LOG_CHANNEL
            or asst._cache.get("LOG_CHANNEL")
            or "me"
        )
        msg = await asst.send_message(
            LOG_CHANNEL, "**Do not delete this file.**", file="database.json"
        )
        asst._cache["TGDB_URL"] = msg.message_link
        udB.set_key("TGDB_URL", msg.message_link)
    except Exception as ex:
        LOGS.error(f"Error on autoupdate_local_database: {ex}")


def update_envs():
    """Update Var. attributes to udB"""
    from .. import udB

    env_keys = list(os.environ.keys())
    if ".env" in os.listdir("."):
        for keys in RepositoryEnv(config._find_file(".")).data:
            env_keys.append(keys)
    for key in env_keys:
        # do not update keys in db from .env file (will toggle this if needed)
        if key in ("LOG_CHANNEL", "BOT_TOKEN", "BOTMODE", "DUAL_MODE", "language"):
            if value := os.environ.get(key):
                udB.set_key(key, value)
            else:
                udB.set_key(key, config.config.get(key))


async def startup_stuff():
    from .. import udB

    for dirs in ("auth", "downloads", "temp"):
        Path("resources").joinpath(dirs).mkdir(parents=True, exist_ok=True)

    if cookie_url := udB.get_key("YT_COOKIES"):
        path = "resources/extras/yt-cookies.txt"
        if not Path(path).is_file():
            try:
                await download_file(cookie_url, path)
            except Exception as er:
                LOGS.exception(er)

    CT = udB.get_key("CUSTOM_THUMBNAIL")
    if CT:
        path = "resources/extras/thumbnail.jpg"
        Path(path).unlink(missing_ok=True)
        try:
            path, _ = await download_file(CT, path)
            ULTConfig.thumb = path
        except Exception as er:
            LOGS.exception(er)
    elif CT is False:
        ULTConfig.thumb = None

    # if GT := udB.get_key("GDRIVE_AUTH_TOKEN"):
    # Path("resources/auth/gdrive_creds.json").write_text(GT)

    udB.del_key("AUTH_TOKEN")

    MM = udB.get_key("MEGA_MAIL")
    MP = udB.get_key("MEGA_PASS")
    if MM and MP:
        Path(".megarc").write_text(f"[Login]\nUsername = {MM}\nPassword = {MP}")


async def _autobot(ultroid_bot, udB):
    if udB.get_key("BOT_TOKEN") or Var.BOT_TOKEN:
        return
    await ultroid_bot.start()
    LOGS.info("MAKING A TELEGRAM BOT FOR YOU AT @BotFather, Kindly Wait")
    who = ultroid_bot.me
    name = who.first_name + "'s Bot"
    if who.username:
        username = who.username + "_bot"
    else:
        username = "ultroid_" + (str(who.id))[5:] + "_bot"
    bf = "@BotFather"
    await ultroid_bot(UnblockRequest(bf))
    await ultroid_bot.send_message(bf, "/cancel")
    await asyncio.sleep(1)
    await ultroid_bot.send_message(bf, "/newbot")
    await asyncio.sleep(1)
    isdone = (await ultroid_bot.get_messages(bf, limit=1))[0].text
    if isdone.startswith("That I cannot do.") or "20 bots" in isdone:
        LOGS.critical(
            "Please make a Bot from @BotFather and add it's token in BOT_TOKEN, as an env var and restart me."
        )
        quit(0)

    await ultroid_bot.send_message(bf, name)
    await asyncio.sleep(1)
    isdone = (await ultroid_bot.get_messages(bf, limit=1))[0].text
    if not isdone.startswith("Good."):
        await ultroid_bot.send_message(bf, "My Assistant Bot")
        await asyncio.sleep(1)
        isdone = (await ultroid_bot.get_messages(bf, limit=1))[0].text
        if not isdone.startswith("Good."):
            LOGS.critical(
                "Please make a Bot from @BotFather and add it's token in BOT_TOKEN, as an env var and restart me."
            )
            quit(0)

    await ultroid_bot.send_message(bf, username)
    await asyncio.sleep(1)
    isdone = (await ultroid_bot.get_messages(bf, limit=1))[0].text
    await ultroid_bot.send_read_acknowledge("botfather")
    if isdone.startswith("Sorry,"):
        ran = randint(1, 100)
        username = "ultroid_" + (str(who.id))[6:] + str(ran) + "_bot"
        await ultroid_bot.send_message(bf, username)
        await asyncio.sleep(1)
        isdone = (await ultroid_bot.get_messages(bf, limit=1))[0].text
    if isdone.startswith("Done!"):
        token = isdone.split("`")[1]
        udB.set_key("BOT_TOKEN", token)
        await _enable_inline(ultroid_bot, username)
        LOGS.info(
            f"Done. Successfully created @{username} to be used as your assistant bot!"
        )
    else:
        LOGS.info(
            "Please Delete Some Of your Telegram bots at @Botfather or Set Var BOT_TOKEN with token of a bot"
        )
        quit(0)


async def autopilot():
    from .. import asst, udB, ultroid_bot

    channel = udB.get_key("LOG_CHANNEL")
    new_channel = None
    if channel:
        try:
            chat = await ultroid_bot.get_entity(channel)
        except BaseException as err:
            LOGS.exception(err)
            udB.del_key("LOG_CHANNEL")
            channel = None
    if not channel:

        async def _save(exc):
            udB._cache["LOG_CHANNEL"] = ultroid_bot.me.id
            await asst.send_message(
                ultroid_bot.me.id, f"Failed to Create Log Channel due to {exc}.."
            )

        if ultroid_bot._bot:
            msg_ = "'LOG_CHANNEL' not found! Add it in order to use 'BOTMODE'"
            LOGS.error(msg_)
            return await _save(msg_)
        LOGS.info("Creating a Log Channel for You!")
        try:
            r = await ultroid_bot(
                CreateChannelRequest(
                    title="My Ultroid Logs",
                    about="My Ultroid Log Group\n\n Join @TeamUltroid",
                    megagroup=True,
                ),
            )
        except ChannelsTooMuchError as er:
            LOGS.critical(
                "You Are in Too Many Channels & Groups , Leave some And Restart The Bot"
            )
            return await _save(str(er))
        except BaseException as er:
            LOGS.exception(er)
            LOGS.info(
                "Something Went Wrong , Create A Group and set its id on config var LOG_CHANNEL."
            )

            return await _save(str(er))
        new_channel = True
        chat = r.chats[0]
        channel = get_peer_id(chat)
        udB.set_key("LOG_CHANNEL", channel)
    assistant = True
    try:
        await ultroid_bot.get_permissions(int(channel), asst.me.username)
    except UserNotParticipantError:
        try:
            await ultroid_bot(InviteToChannelRequest(int(channel), [asst.me.username]))
        except BaseException as er:
            LOGS.info("Error while Adding Assistant to Log Channel")
            LOGS.exception(er)
            assistant = False
    except BaseException as er:
        assistant = False
        LOGS.exception(er)
    if assistant and new_channel:
        try:
            achat = await asst.get_entity(int(channel))
        except BaseException as er:
            achat = None
            LOGS.info("Error while getting Log channel from Assistant")
            LOGS.exception(er)
        if achat and not achat.admin_rights:
            rights = ChatAdminRights(
                add_admins=True,
                invite_users=True,
                change_info=True,
                ban_users=True,
                delete_messages=True,
                pin_messages=True,
                anonymous=False,
                manage_call=True,
            )
            try:
                await ultroid_bot(
                    EditAdminRequest(
                        int(channel), asst.me.username, rights, "Assistant"
                    )
                )
            except ChatAdminRequiredError:
                LOGS.info(
                    "Failed to promote 'Assistant Bot' in 'Log Channel' due to 'Admin Privileges'"
                )
            except BaseException as er:
                LOGS.info("Error while promoting assistant in Log Channel..")
                LOGS.exception(er)
    if isinstance(chat.photo, ChatPhotoEmpty):
        photo, _ = await download_file(
            "https://graph.org/file/27c6812becf6f376cbb10.jpg", "channelphoto.jpg"
        )
        ll = await ultroid_bot.upload_file(photo)
        try:
            await ultroid_bot(
                EditPhotoRequest(int(channel), InputChatUploadedPhoto(ll))
            )
        except BaseException as er:
            LOGS.exception(er)
        os.remove(photo)


# customize assistant
async def customize():
    from .. import asst, udB, ultroid_bot

    rem = None
    try:
        chat_id = udB.get_key("LOG_CHANNEL")
        if asst.me.photo:
            return
        LOGS.info("Customising Ur Assistant Bot in @BOTFATHER")
        UL = f"@{asst.me.username}"
        if not ultroid_bot.me.username:
            sir = ultroid_bot.me.first_name
        else:
            sir = f"@{ultroid_bot.me.username}"
        file = random.choice(
            [
                "https://graph.org/file/92cd6dbd34b0d1d73a0da.jpg",
                "https://graph.org/file/a97973ee0425b523cdc28.jpg",
                "resources/extras/ultroid_assistant.jpg",
            ]
        )
        if not os.path.exists(file):
            file, _ = await download_file(file, "profile.jpg")
            rem = True
        msg = await asst.send_message(
            chat_id, "**Auto Customisation** Started on @Botfather"
        )
        await asyncio.sleep(1)
        await ultroid_bot.send_message("botfather", "/cancel")
        await asyncio.sleep(1)
        await ultroid_bot.send_message("botfather", "/setuserpic")
        await asyncio.sleep(1)
        isdone = (await ultroid_bot.get_messages("botfather", limit=1))[0].text
        if isdone.startswith("Invalid bot"):
            LOGS.info("Error while trying to customise assistant, skipping...")
            return
        await ultroid_bot.send_message("botfather", UL)
        await asyncio.sleep(1)
        await ultroid_bot.send_file("botfather", file)
        await asyncio.sleep(2)
        await ultroid_bot.send_message("botfather", "/setabouttext")
        await asyncio.sleep(1)
        await ultroid_bot.send_message("botfather", UL)
        await asyncio.sleep(1)
        await ultroid_bot.send_message(
            "botfather", f"✨ Hello ✨!! I'm Assistant Bot of {sir}"
        )
        await asyncio.sleep(2)
        await ultroid_bot.send_message("botfather", "/setdescription")
        await asyncio.sleep(1)
        await ultroid_bot.send_message("botfather", UL)
        await asyncio.sleep(1)
        await ultroid_bot.send_message(
            "botfather",
            f"✨ Powerful Ultroid Assistant Bot ✨\n✨ Master ~ {sir} ✨\n\n✨ Powered By ~ @TeamUltroid ✨",
        )
        await asyncio.sleep(2)
        await msg.edit("Completed **Auto Customisation** at @BotFather.")
        if rem:
            os.remove(file)
        LOGS.info("Customisation Done")
    except Exception as e:
        LOGS.exception(e)


"""
async def plug_unzipper(ult, chat):
    from shutil import rmtree
    from .utils import load_addons
    from .. import Var

    moimsg = await ult.get_messages(
        chat, search="PLUGIN_SOURCE.zip", filter=InputMessagesFilterDocument, limit=1
    )
    if not moimsg:
        return True
    x = moimsg[0]
    if x and x.file.name == "PLUGIN_SOURCE.zip":
        os.mkdir("tplugs")
        temp = await x.download_media("source.zip")
    else:
        return True

    await bash("unzip -q source.zip -d tplugs")
    plen = len(os.listdir("tplugs/t1"))
    os.remove("source.zip")
    LOGS.info(f"{'•'*4} {chat} || Installing {plen} Plugins!")
    for file in os.scandir("tplugs/t1"):
        plugin = file.name
        _path = os.path.join("addons", plugin)
        if os.path.isfile(_path):
            if Var.HOST.lower() not in ("local", "railway"):
                LOGS.warning(f"Plugin {plugin} already Exists in Addons folder.")
            continue
        try:
            os.rename(file.path, _path)
            load_addons(_path)
        except BaseException as exc:
            os.remove(_path)
            LOGS.info(f"Ultroid - PLUGIN_CHANNEL - ERROR - {plugin}")
            LOGS.exception(exc)
    rmtree("tplugs")
"""


async def plug(plugin_channels):
    from .. import ultroid_bot
    from .utils import load_addons

    if ultroid_bot._bot:
        LOGS.info("Plugin Channels can't be used in 'BOTMODE'")
        return
    if os.path.exists("addons") and not os.path.exists("addons/.git"):
        shutil.rmtree("addons")
    Path("addons").mkdir(exist_ok=True)
    if not os.path.exists("addons/__init__.py"):
        Path("addons/__init__.py").write_text(
            "from plugins import *\n\nbot = ultroid_bot"
        )
    LOGS.info("• Loading Plugins from Plugin Channel(s) •")
    for chat in plugin_channels:
        # plugUnzippr = await plug_unzipper(ultroid_bot, chat)
        # if not plugUnzippr: continue
        try:
            LOGS.info(f"{'•' * 4} {chat}")
            async for x in ultroid_bot.iter_messages(
                chat, search=".py", filter=InputMessagesFilterDocument, wait_time=10
            ):
                if x.text == "#IGNORE":
                    continue
                plugin = "addons/" + x.file.name.replace("_", "-").replace("|", "-")
                if not os.path.exists(plugin):
                    await asyncio.sleep(0.6)
                    plugin = await x.download_media(plugin)
                    try:
                        load_addons(plugin)
                    except Exception as e:
                        LOGS.exception(f"Ultroid - PLUGIN_CHANNEL - ERROR - {plugin}")
                        os.remove(plugin)
        except Exception as er:
            LOGS.exception(er)


# some stuffs


async def fetch_ann():
    from .. import asst, udB
    from ..fns.tools import async_searcher

    get_ = udB.get_key("OLDANN") or []
    chat_id = udB.get_key("LOG_CHANNEL")

    try:
        updts = await async_searcher(
            "https://ultroid-api.vercel.app/announcements",
            post=True,
            re_json=True,
        )
        for upt in updts:
            key = list(upt.keys())[0]
            if key not in get_:
                cont = upt[key]
                if isinstance(cont, dict) and cont.get("lang"):
                    if cont["lang"] != (udB.get_key("language") or "en"):
                        continue
                    cont = cont["msg"]
                if isinstance(cont, str):
                    await asst.send_message(chat_id, cont)
                elif isinstance(cont, dict) and cont.get("chat"):
                    await asst.forward_messages(chat_id, cont["msg_id"], cont["chat"])
                else:
                    LOGS.info(cont)
                    LOGS.info(
                        "Invalid Type of Announcement Detected!\nMake sure you are on latest version.."
                    )
                get_.append(key)
        udB.set_key("OLDANN", get_)
    except Exception as er:
        LOGS.exception(er)


async def WasItRestart(udb):
    key = udb.get_key("_RESTART")
    if not key:
        return
    from .. import asst, ultroid_bot

    try:
        data = key.split("_")
        who = asst if data[0] == "bot" else ultroid_bot
        await who.edit_message(
            int(data[1]), int(data[2]), "__Restarted Successfully.__"
        )
    except Exception as er:
        LOGS.exception("Restart Message Edit Error")
    finally:
        udb.del_key("_RESTART")


async def ready():
    from re import purge as re_purge
    from .. import asst, udB, ultroid_bot

    chat_id = udB.get_key("LOG_CHANNEL")
    spam_sent, PHOTO, BTTS = None, None, None
    if not udB.get_key("INIT_DEPLOY"):
        # Detailed Message at Initial Deploy
        MSG = """🎇 **Thanks for Deploying Ultroid Userbot!**
• Here, are the Some Basic stuff from, where you can Know, about its Usage."""
        PHOTO = "https://graph.org/file/54a917cc9dbb94733ea5f.jpg"
        BTTS = Button.inline("• Click to Start •", "initft_2")
        udB.set_key("INIT_DEPLOY", "Done")
    else:
        MSG = f"**Ultroid has been deployed!**\n➖➖➖➖➖➖➖➖➖➖\n**UserMode**: {inline_mention(ultroid_bot.me)}\n**Assistant**: @{asst.me.username}\n➖➖➖➖➖➖➖➖➖➖\n**Support**: @TeamUltroid\n➖➖➖➖➖➖➖➖➖➖"
        updt, _ = await custom_updater()
        if updt:
            BTTS = Button.inline("Update Available", "updtavail")

        """
        prev_spam = udB.get_key("LAST_UPDATE_LOG_SPAM")
        if prev_spam:
            try:
                await ultroid_bot.delete_messages(chat_id, int(prev_spam))
            except Exception as E:
                LOGS.info("Error while Deleting Previous Update Message :" + str(E))
        """

    try:
        spam_sent = await asst.send_message(chat_id, MSG, file=PHOTO, buttons=BTTS)
    except ValueError as e:
        try:
            await (await ultroid_bot.send_message(chat_id, str(e))).delete()
            spam_sent = await asst.send_message(chat_id, MSG, file=PHOTO, buttons=BTTS)
        except Exception as g:
            spam_sent = None
            LOGS.error(g)
    except Exception as el:
        LOGS.error(el)
        try:
            spam_sent = await ultroid_bot.send_message(chat_id, MSG)
        except Exception as ef:
            spam_sent = None
            LOGS.error(ef)

    # if spam_sent and not spam_sent.media:
    # udB.set_key("LAST_UPDATE_LOG_SPAM", spam_sent.id)

    re_purge()
    try:
        await WasItRestart(udB)
        # await fetch_ann()
    except Exception as exc:
        LOGS.exception(exc)


def _version_changes(udb):
    for _ in (
        "BOT_USERS",
        "BOT_BLS",
        "VC_SUDOS",
        "SUDOS",
        "CLEANCHAT",
        "LOGUSERS",
        "PLUGIN_CHANNEL",
        "CH_SOURCE",
        "CH_DESTINATION",
        "BROADCAST",
    ):
        key = udb.get_key(_)
        if key and str(key)[0] != "[":
            key = udb.get(_)
            new_ = [
                int(z) if z.isdigit() or (z.startswith("-") and z[1:].isdigit()) else z
                for z in key.split()
            ]
            udb.set_key(_, new_)


async def _enable_inline(ultroid_bot, username):
    bf = "BotFather"
    await ultroid_bot.send_message(bf, "/setinline")
    await asyncio.sleep(1)
    await ultroid_bot.send_message(bf, f"@{username}")
    await asyncio.sleep(1)
    await ultroid_bot.send_message(bf, "Search")
    await ultroid_bot.send_read_acknowledge(bf)


__all__ = (
    "WasItRestart",
    "autopilot",
    "customize",
    "plug",
    "ready",
    "startup_stuff",
    "update_envs",
)

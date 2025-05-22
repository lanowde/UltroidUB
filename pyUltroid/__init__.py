# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.


import time


start_time = time.time()
_ignore_eval = []
_shutdown_tasks = []
_ult_cache = {}


try:
    import telethonpatch as _telepatch
except ImportError:
    # print("running without telethonpatch >>>")
    pass


from .version import __version__, ultroid_version
from .custom.init import loop
from .configs import Var
from .startup import *


class ULTConfig:
    lang = "en"
    thumb = "resources/extras/ultroid.jpg"


# database stuff
from .startup._database import _UltroidDB

udB = _UltroidDB()
LOGS.info(f"Connected to {udB.name} Successfully!")


from .custom.init import afterBoot as _afterBoot
from .startup.BaseClient import UltroidClient as _UltroidClient
from .startup.connections import (
    validate_session as _validate_session,
    vc_connection as _vc_connection,
)
from .startup.funcs import _autobot, _enable_inline


_afterBoot(udB)
BOT_MODE = udB.get_key("BOTMODE")
DUAL_MODE = udB.get_key("DUAL_MODE")

USER_MODE = udB.get_key("USER_MODE")
if USER_MODE:
    DUAL_MODE = False

if BOT_MODE:
    if DUAL_MODE:
        udB.del_key("DUAL_MODE")
        DUAL_MODE = False
    ultroid_bot = None

    if not (udB.get_key("BOT_TOKEN") or Var.BOT_TOKEN):
        LOGS.critical('"BOT_TOKEN" not Found! Please add it, in order to use "BOTMODE"')
        quit(1)
else:
    ultroid_bot = _UltroidClient(
        _validate_session(Var.SESSION, LOGS),
        udB=udB,
        app_version=ultroid_version,
        device_model="Ultroid",
        proxy=udB.get_key("TG_PROXY"),
        entity_cache_limit=2500,
    )
    ultroid_bot.run_in_loop(_autobot(ultroid_bot, udB))

if USER_MODE:
    asst = ultroid_bot
else:
    asst = _UltroidClient(
        None,
        bot_token=udB.get_key("BOT_TOKEN") or Var.BOT_TOKEN,
        udB=udB,
        entity_cache_limit=500,
    )

if BOT_MODE:
    ultroid_bot = asst
    if udB.get_key("OWNER_ID"):
        try:
            ultroid_bot.me = ultroid_bot.run_in_loop(
                ultroid_bot.get_entity(udB.get_key("OWNER_ID"))
            )
        except Exception as er:
            LOGS.exception(er)
elif not asst.me.bot_inline_placeholder:
    ultroid_bot.run_in_loop(_enable_inline(ultroid_bot, asst.me.username))


vcClient = _vc_connection(udB, ultroid_bot)


HNDLR = udB.get_key("HNDLR") or "."
DUAL_HNDLR = udB.get_key("DUAL_HNDLR") or "/"
SUDO_HNDLR = udB.get_key("SUDO_HNDLR") or HNDLR


from .custom.startup_helper import _handle_post_startup

_handle_post_startup()

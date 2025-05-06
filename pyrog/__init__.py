# @spemgod | @ah3h3 | @moiusrname
#
# This is just for personal use,
# cz telethon is quite slow in transferring files.
#
# Edited on -
# 13-05-2022 // for pyrogram v2.
# 28-06-2022 // added asyncio.gather for faster startup.
# 22-05-2023 // slower startup ~ clients starting one by one in background.
# 15-01-2024 // removed plugin support.

import asyncio
import os
from ast import literal_eval
from copy import deepcopy

from pyUltroid import LOGS, _shutdown_tasks
from pyUltroid.configs import Var
from pyUltroid.custom.init import run_async_task

try:
    from pyrogram import Client
except ImportError:
    Client = None


_error = ""  # to check if there are errors
PYROG_CLIENTS = {}
_workers = 2 if os.cpu_count() < 3 else min(6, os.cpu_count())

_default_client_values = {
    "api_id": Var.API_ID,
    "api_hash": Var.API_HASH,
    "workdir": "resources/auth/",
    "sleep_threshold": 300,
    "workers": _workers,
    "no_updates": True,
    "max_concurrent_transmissions": 1,
    "max_message_cache_size": 512,  # kurigram
    "max_topic_cache_size": 128,
}


def app(n=None):
    _default = PYROG_CLIENTS.get(1)
    return PYROG_CLIENTS.get(n, _default) if n else _default


def setup_clients():
    # plugins = {"root": "pyrog/plugins"}
    if not Client:
        _error = "'pyrogram' is not installed; Skipping pyrogram setup.."
        return True

    var = "PYROGRAM_CLIENTS"
    stuff = os.environ.get(var)
    if not stuff:
        _error = "Var 'PYROGRAM_CLIENTS' not found; Skipping pyrogram setup.."
        return True

    data = literal_eval(stuff)
    if Var.HOST.lower() == "heroku":
        os.environ.pop(var, None)
    for k, v in data.items():
        _default = deepcopy(_default_client_values)
        _default["name"] = "bot_" + str(k)
        _default["bot_token"] = v.get("bot_token") if type(v) == dict else v
        PYROG_CLIENTS[int(k)] = Client(**_default)


async def pyro_startup():
    if setup_clients():
        return LOGS.warning(_error)

    LOGS.info("Starting Pyrogram...")
    for count, client in PYROG_CLIENTS.copy().items():
        try:
            await client.start()
        except Exception:
            LOGS.warning(f"Error while starting PyroGram Client: {count}")
            LOGS.debug("error:", exc_info=True)
            PYROG_CLIENTS.pop(count, None)
        else:
            _shutdown_tasks.append(client.stop())
        finally:
            await asyncio.sleep(2)

    LOGS.info(
        f"{len(PYROG_CLIENTS)} Pyrogram Clients Running -> {tuple(PYROG_CLIENTS.keys())}"
    )


async def _init_pyrog():
    run_async_task(pyro_startup, id="pyrogram_startup")
    if not _error:
        await asyncio.sleep(9)

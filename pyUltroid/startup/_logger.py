__all__ = (
    "LOGS",
    "HOSED_ON",
    "TelethonLogger",
    "where_hosted",
    "LOG_HANDLERS",
)

import logging
from ast import literal_eval
from os import environ
from platform import python_version, python_version_tuple

from telethon import __version__
from telethon.tl.alltlobjects import LAYER

from pyUltroid.configs import Var
from ._extra import _fix_logging, _ask_input
from pyUltroid.version import ultroid_version, __version__ as __pyUltroid__


# ----------------------------------------------------------------------------

LOG_DATA = {}
LOG_HANDLERS = []

LOGS = logging.getLogger("pyUltLogs")
LOGS.setLevel(logging.DEBUG)

# ----------------------------------------------------------------------------


def where_hosted():
    if environ.get("DYNO"):
        return "heroku"
    elif environ.get("RAILWAY_STATIC_URL"):
        return "railway"
    elif environ.get("OKTETO_TOKEN"):
        return "okteto"
    elif environ.get("RUNNER_USER") or environ.get("HOSTNAME"):
        if environ.get("USER") == "codespace":
            return "codespace"
        return "github actions"
    elif environ.get("KUBERNETES_PORT"):
        return "qovery | kubernetes"
    elif environ.get("ANDROID_ROOT"):
        return "termux"
    elif environ.get("FLY_APP_NAME"):
        return "fly.io"
    return "local"


HOSTED_ON = where_hosted()

# ----------------------------------------------------------------------------

if int(python_version_tuple()[1]) < 10:
    _fix_logging(logging.FileHandler)

if HOSTED_ON == "local" or environ.get("HOST") == "local":
    _ask_input()

if data := environ.get("LOGGER_DATA"):
    LOG_DATA = literal_eval(data)
    if Var.HOST.lower() == "heroku":
        environ.pop("LOGGER_DATA", None)

# ----------------------------------------------------------------------------

TelethonLogger = logging.getLogger("Telethon")
TelethonLogger.setLevel(
    logging.INFO if LOG_DATA.get("verbose") is True else logging.WARNING
)

default_format = "%(asctime)s | %(name)s [%(levelname)s] : %(message)s"
default_formatter = logging.Formatter(default_format, datefmt="%m/%d, %H:%M:%S")

logging.getLogger("pyrogram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("pyrogram.parser.html").setLevel(logging.ERROR)
logging.getLogger("pyrogram.session.session").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------


def setup_log_handlers():
    kwargs = dict(
        filename="ultlogs.txt",
        mode="w",
        encoding="utf-8",
        errors="ignore",
    )
    if int(python_version_tuple()[1]) < 9:
        # unsupported on python 3.8 or lower
        kwargs.pop("errors", None)

    file_handler = logging.FileHandler(**kwargs)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(default_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(default_formatter)

    LOG_HANDLERS.extend((file_handler, stream_handler))


setup_log_handlers()

# ----------------------------------------------------------------------------


def setup_tglogger():
    if LOG_DATA.get("pyro") in (True, "True"):
        from pyUltroid.custom.tglogger import PyroTGLogHandler as TGLogger
    else:
        from pyUltroid.custom.tglogger import TGLogHandlerBotAPI as TGLogger

    tglogger = TGLogger(chat=LOG_DATA.get("chat"), token=LOG_DATA.get("token"))
    uname = LOG_DATA.get("name", "TGLogger")
    tglogger.setLevel(logging.DEBUG)
    tglogger.setFormatter(
        logging.Formatter(
            uname
            + " [%(levelname)s] (%(asctime)s)\n» Line %(lineno)s: %(filename)s\n» %(message)s",
            datefmt="%d %b %H:%M:%S",
        )
    )
    LOG_HANDLERS.append(tglogger)


if LOG_DATA.get("tglog") in (True, "True"):
    setup_tglogger()

# ----------------------------------------------------------------------------

# Initiate all Loggers!
logging.basicConfig(handlers=LOG_HANDLERS)
del LOG_DATA, LOG_HANDLERS

if Var.HOST.lower() == "local":
    try:
        import coloredlogs

        coloredlogs.install(level=None, logger=LOGS, fmt=default_format)
    except ImportError:
        pass

# ----------------------------------------------------------------------------

LOGS.info(
    f"""
                ------------------------------------
                       Starting Deployment!
                ------------------------------------
"""
)

LOGS.info(f"Python version - {python_version()}")
LOGS.info(f"py-Ultroid Version - {__pyUltroid__}")
LOGS.info(f"Telethon Version - {__version__} [Layer: {LAYER}]")
LOGS.info(f"Ultroid Version - {ultroid_version} [{HOSTED_ON}]")

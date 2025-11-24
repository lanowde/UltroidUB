# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

__all__ = ("validate_session", "vc_connection")

import base64
import ipaddress
import struct
import sys

from telethon.errors.rpcerrorlist import AuthKeyDuplicatedError
from telethon.sessions.string import _STRUCT_PREFORMAT, CURRENT_VERSION, StringSession

from strings import get_string
from pyUltroid.configs import Var
from . import *

try:
    from telethon.sessions.asqlite import AsyncSQLite
except ImportError:
    AsyncSQLite = None


_PYRO_FORM = {351: ">B?256sI?", 356: ">B?256sQ?", 362: ">BI?256sQ?"}

# https://github.com/pyrogram/pyrogram/blob/master/docs/source/faq/what-are-the-ip-addresses-of-telegram-data-centers.rst
DC_IPV4 = {
    1: "149.154.175.53",
    2: "149.154.167.51",
    3: "149.154.175.100",
    4: "149.154.167.91",
    5: "91.108.56.130",
}


def validate_session(session, logger=LOGS, _exit=True):
    from strings import get_string

    if session:
        # Telethon Session
        if session.startswith(CURRENT_VERSION):
            if len(session.strip()) != 353:
                logger.exception(get_string("py_c1"))
                sys.exit()
            return StringSession(session)

        # Pyrogram Session
        elif len(session) in _PYRO_FORM.keys():
            data_ = struct.unpack(
                _PYRO_FORM[len(session)],
                base64.urlsafe_b64decode(session + "=" * (-len(session) % 4)),
            )
            if len(session) in (351, 356):
                auth_id = 2
            else:
                auth_id = 3
            dc_id, auth_key = data_[0], data_[auth_id]
            return StringSession(
                CURRENT_VERSION
                + base64.urlsafe_b64encode(
                    struct.pack(
                        _STRUCT_PREFORMAT.format(4),
                        dc_id,
                        ipaddress.ip_address(DC_IPV4[dc_id]).packed,
                        443,
                        auth_key,
                    )
                ).decode("ascii")
            )
        else:
            logger.exception(get_string("py_c1"))
            if _exit:
                sys.exit()
    logger.exception(get_string("py_c2"))
    if _exit:
        sys.exit()


def vc_connection(udB, ultroid_bot):
    from .BaseClient import UltroidClient

    VC_SESSION = Var.VC_SESSION or udB.get_key("VC_SESSION")
    if VC_SESSION and VC_SESSION != Var.SESSION:
        LOGS.info("Starting Seperate VcClient..")
        try:
            vc_client = UltroidClient(
                validate_session(VC_SESSION, _exit=False),
                log_attempt=False,
                exit_on_error=False,
            )
            LOGS.info("Successfully Started VcClient!")
            return vc_client
        except (AuthKeyDuplicatedError, EOFError):
            LOGS.info(get_string("py_c3"))
            udB.del_key("VC_SESSION")
        except Exception:
            LOGS.exception("Error While starting vcClient")
    return ultroid_bot


# todo: check if it even works.
def connect_ub(s):
    from .BaseClient import UltroidClient

    VC_SESSION = Var.VC_SESSION or udB.get_key("VC_SESSION")
    if s and s not in (VC_SESSION, Var.SESSION):
        LOGS.debug("tying to boot up new Client")
        try:
            return UltroidClient(
                validate_session(s, _exit=False),
                exit_on_error=False,
            )
        except (AuthKeyDuplicatedError, EOFError) as er:
            LOGS.exception("Error in the new Client.!")
            return er
        except Exception as er:
            LOGS.exception("Error while creating new Client.")


def init_session(ub=False, asst=False):
    key = udB.get_key("_PREFER_ASYNC_SQLITE")
    if ub:
        return (
            AsyncSQLite(key)
            if key and AsyncSQLite and HOSTED_ON == "termux"
            else validate_session(Var.SESSION, LOGS)
        )
    elif asst:
        return (
            AsyncSQLite("asst")
            if key and AsyncSQLite and HOSTED_ON == "termux"
            else None
        )

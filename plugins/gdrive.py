# Ultroid - UserBot
# Copyright (C) 2020-2023 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

"""
✘ Commands Available -

• `{i}gdown <gdrive url>`

• `{i}gup <reply to file/path>`
  Flags: (k) to use other account!
  Eg: `{i}gup -k=2` to use 2nd GDrive account.

•• Required KEYS for gdrive uploader:
- `GDRIVE_CLIENT_ID`
- `GDRIVE_CLIENT_SECRET`
- `GDRIVE_FOLDER_ID`  <optional>
- `GDRIVE_CLIENT_ID_2`  <for 2nd account>

•• You also need to Authorise GDrive once from /start menu of Assistant, which automatically sets `GDRIVE_AUTH_TOKEN` !
"""

import asyncio
import os

from pyUltroid.fns.gDrive import GDriveManager

from . import (
    LOGS,
    get_string,
    tg_downloader,
    time_formatter,
    udB,
    ultroid_cmd,
    unix_parser,
)


@ultroid_cmd(
    pattern="gup( (.*)|$)",
)
async def gdrive_uploader(event):
    reply = await event.get_reply_message()
    inpt = event.pattern_match.group(2)

    if not (reply or inpt):
        return await event.eor("`Reply to file or give its path to upload to Gdrive!`")

    args = unix_parser(input_file or "")
    key_suffix = args.kwargs.get("k")
    filename = args.args

    GD = GDriveManager(key_suffix)
    if not GD.creds:
        return await event.eor(
            "`Credentials have not been added; add GDrive tokens to Use it..`"
        )

    mone = await event.eor(get_string("com_1"))
    if reply:
        try:
            filename, t_time = await tg_downloader(
                media=reply,
                event=mone,
                show_progress=True,
            )
        except Exception as exc:
            LOGS.exception(exc)
            return await mone.edit(f"**Error in downloading file:** `{exc}`")

        tt = time_formatter(t_time * 1000)
        await mone.edit(f"Downloaded in {tt}.\n – `{filename}`\n")
        await asyncio.sleep(2)

    if not (os.path.isfile(filename) and os.path.getsize(filename) != 0):
        return await mone.eor(
            "`File Not found in local server or File Size is 0B.\n\n Give me a valid file path :((`",
        )

    try:
        m_time = time.time()
        resp = await GD.upload_file(mone, filename)
        await asyncio.sleep(1)
        text = "**GDrive Upload was Successful!** \n\n**File Name:** `{name}` \n**Size:** `{size}` \n**GDrive Link:** [Click Here]({link}) \n**Time Taken:** `{time_taken}`"
        await mone.edit(
            text.format(
                name=resp.get("name"),
                size=humanbytes(os.path.getsize(filename)),
                link=resp.get("webContentLink"),
                time_taken=time_formatter((time.time() - m_time) * 1000),
            )
        )
    except Exception as exc:
        LOGS.exception(exc)
        await mone.edit(f"Exception occurred while uploading to gDrive: \n`{exc}`")


@ultroid_cmd(
    pattern="gdown( (.*)|$)",
)
async def gdrive_downloader(e):
    match = e.pattern_match.group(2)
    if not match:
        return await e.eor("`Give GDrive Link to Download from..`")

    args = unix_parser(match or "")
    key_suffix = args.kwargs.get("k")
    match = args.args

    GD = GDriveManager(key_suffix)
    if not GD.creds:
        return await e.eor(
            "`Credentials have not been added; add GDrive tokens to Use it..`"
        )

    if not GD.extract_drive_id(match):
        return await e.eor("`This link seems to be invalid GDrive url..`")

    eve = await e.eor(get_string("com_1"))
    _start = time.time()
    status, response = await GD.download_file(eve, match)
    if not status:
        LOGS.exception(response)
        return await eve.edit(f"`Exception occurred while Downloading from gDrive...`")

    source = f"[GDrive]({match})" if "https" in match else "GDrive"
    size = humanbytes(os.path.getsize(response))
    time_taken = time_formatter((time.time() - _start) * 1000)
    text = f"**GDrive Download was Successful!** \n\nPath: `{response}` \nSize: `{size}` \nSource: {source} \nTime Taken: `{time_taken}`"
    await eve.edit(text)

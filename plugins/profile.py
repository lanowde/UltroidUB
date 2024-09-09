# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

"""
✘ Commands Available -

• `{i}setname <first name // last name>`
    Change your profile name.

• `{i}setbio <bio>`
    Change your profile bio.

• `{i}setpic <reply to pic>`
    Change your profile pic.

• `{i}delpfp <n>(optional)`
    Delete one profile pic, if no value given, else delete n number of pics.

• `{i}poto <username>/reply`
  `{i}poto <reply/upload-limit>/all`

  Ex: `{i}poto 10` - uploads starting 10 pfps of user.
    Upload the photo of Chat/User if Available.
"""

import asyncio

from telethon.tl.functions.account import UpdateProfileRequest
from telethon.tl.functions.photos import DeletePhotosRequest, UploadProfilePhotoRequest

from . import get_string, mediainfo, ultroid_cmd, osremove


# bio changer
@ultroid_cmd(pattern="setbio( (.*)|$)", fullsudo=True)
async def setbio_(ult):
    ok = await ult.eor("...")
    set = ult.pattern_match.group(1).strip()
    try:
        await ult.client(UpdateProfileRequest(about=set))
        await ok.eor(f"Profile bio changed to - `{set}`", time=6)
    except Exception as ex:
        await ok.edit(f"**Error occured.**\n`{str(ex)}`")


# name changer
@ultroid_cmd(pattern="setname ?((.|//)*)", fullsudo=True)
async def setname_(ult):
    ok = await ult.eor("...")
    names = first_name = ult.pattern_match.group(2)
    last_name = ""
    if "//" in names:
        first_name, last_name = names.split("//", 1)
    try:
        await ult.client(
            UpdateProfileRequest(
                first_name=first_name,
                last_name=last_name,
            ),
        )
        await ok.eor(f"Name changed to `{names}`", time=10)
    except Exception as ex:
        await ok.edit(f"Error occured.\n`{str(ex)}`")


# profile pic
@ultroid_cmd(pattern="setpic$", fullsudo=True)
async def setpic_(ult):
    if not ult.is_reply:
        return await ult.eor("`Reply to a Media..`", time=5)
    reply_message = await ult.get_reply_message()
    ok = await ult.eor(get_string("com_1"))
    replfile = await reply_message.download_media()
    file = await ult.client.upload_file(replfile)
    try:
        if "pic" in mediainfo(reply_message.media):
            await ult.client(UploadProfilePhotoRequest(file=file))
        else:
            await ult.client(UploadProfilePhotoRequest(video=file))
        osremove(replfile)
        await ok.edit("`My Profile Photo has Successfully Changed !`")
    except Exception as ex:
        await ok.edit(f"**Error occured.**\n`{str(ex)}`")


# delete profile pic(s)
@ultroid_cmd(pattern="delpfp( (.*)|$)", fullsudo=True)
async def remove_profilepic(delpfp):
    ok = await delpfp.eor("`...`")
    group = delpfp.pattern_match.group(2)
    if group == "all":
        lim = 0
    elif group and group.isdigit():
        lim = int(group)
    else:
        lim = 1
    pfplist = await delpfp.client.get_profile_photos("me", limit=lim)
    await delpfp.client(DeletePhotosRequest(pfplist))
    await ok.edit(f"`Successfully deleted {len(pfplist)} profile picture(s).`")


@ultroid_cmd(pattern="poto( (.*)|$)")
async def gpoto(e):
    inpt = e.pattern_match.group(2)
    limit = 1
    if e.reply_to:
        gs = await e.get_reply_message()
        user_id = gs.sender_id
        limit = inpt if inpt else None
    elif inpt:
        split = inpt.split(maxsplit=1)
        if len(split) > 1:
            user_id, limit = split
        else:
            user_id = inpt
        try:
            user_id = int(user_id)
        except ValueError:
            pass
    else:
        user_id = e.chat_id

    a = await e.eor(get_string("com_1"))
    okla = []
    if limit == "all":
        limit = None
    else:
        try:
            limit = int(limit)
        except ValueError:
            limit = 1

    if limit == 1 or e.client._bot:
        okla.append(await e.client.download_profile_photo(user_id))
    else:
        async for photo in e.client.iter_profile_photos(user_id, limit=limit):
            photo_path = await e.client.download_media(photo)
            if photo.video_sizes:
                await e.respond(f"`{user_id}`", file=photo_path)
                osremove(photo_path)
                await asyncio.sleep(3)
            else:
                okla.append(photo_path)

    if not okla or okla[0] == None:
        return await a.eor(f"`Pfp Not Found for {user_id}...`", time=8)

    await e.respond(f"`{user_id}`", file=okla)
    osremove(*okla)
    await a.edit(f"`Uploaded {len(okla)} pfp(s) of {user_id}!`")

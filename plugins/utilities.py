# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

"""
✘ Commands Available -

• `{i}kickme` : Leaves the group.

• `{i}date` : Show Calender.

• `{i}listreserved`
    List all usernames (channels/groups) you own.

• `{i}stats` : See your profile stats.

• `{i}paste` - `Include long text / Reply to text file.`

• `{i}info <username/userid/chatid>`
    Reply to someone's msg.

• `{i}invite <username/userid>`
    Add user to the chat.

• `{i}rmbg <reply to pic>`
    Remove background from that picture.

• `{i}telegraph <reply to media/text>`
    Upload media/text to telegraph.

• `{i}json <reply to msg>`
    Get the json encoding of the message.

• `{i}suggest <reply to message> or <poll title>`
    Create a Yes/No poll for the replied suggestion.

• `{i}ipinfo <ipAddress>` : Get info about that IP address.

• `{i}cpy <reply to message>`
   Copy the replied message, with formatting. Expires in 24hrs.
• `{i}pst`
   Paste the copied message, with formatting.

• `{i}thumb <reply file>` : Download the thumbnail of the replied file.

• `{i}getmsg <message link>`
  Get messages from chats with forward/copy restrictions.
"""

import asyncio
import calendar
import html
import io
import os
import pathlib
import time
from datetime import datetime as dt
from shlex import quote as shquote

from telethon.utils import get_peer_id
from telethon.errors.rpcerrorlist import ChatForwardsRestrictedError, UserBotError
from telethon.events import NewMessage
from telethon.tl.custom import Dialog
from telethon.tl.functions.channels import (
    GetAdminedPublicChannelsRequest,
    InviteToChannelRequest,
    LeaveChannelRequest,
)
from telethon.tl.functions.contacts import GetBlockedRequest
from telethon.tl.functions.messages import AddChatUserRequest, GetAllStickersRequest
from telethon.tl.functions.users import GetFullUserRequest
from telethon.tl.types import (
    Channel,
    Chat,
    InputMediaPoll,
    TextWithEntities,
    Poll,
    PollAnswer,
    TLObject,
    User,
)

from pyUltroid._misc._assistant import asst_cmd
from pyUltroid.dB.gban_mute_db import is_gbanned
from pyUltroid.fns.tools import get_chat_and_msgid
from pyUltroid.fns.info import get_chat_info
from pyUltroid.fns.misc import ReTrieveFile
from . import (
    Catbox,
    HNDLR,
    LOGS,
    TelegraphClient,
    asst,
    asyncread,
    async_searcher,
    bash,
    check_filename,
    eod,
    eor,
    get_paste,
    get_string,
    inline_mention,
    json_parser,
    mediainfo,
    osremove,
    tg_downloader,
    udB,
    ultroid_cmd,
)

try:
    from PIL import Image
except ImportError:
    Image = None


# =================================================================#


TMP_DOWNLOAD_DIRECTORY = "resources/downloads/"

_copied_msg = {}


@ultroid_cmd(pattern="kickme$", fullsudo=True)
async def leave(ult):
    await ult.eor(f"`{ult.client.me.first_name} has left this group, bye!!.`")
    await ult.client(LeaveChannelRequest(ult.chat_id))


@ultroid_cmd(
    pattern="date$",
)
async def date(event):
    m = dt.now().month
    y = dt.now().year
    d = dt.now().strftime("Date - %B %d, %Y\nTime- %H:%M:%S")
    k = calendar.month(y, m)
    await event.eor(f"`{k}\n\n{d}`")


@ultroid_cmd(
    pattern="listreserved$",
)
async def _(event):
    result = await event.client(GetAdminedPublicChannelsRequest())
    if not result.chats:
        return await event.eor("`No username Reserved`")
    output_str = "".join(
        f"- {channel_obj.title} @{channel_obj.username} \n"
        for channel_obj in result.chats
    )
    await event.eor(output_str)


@ultroid_cmd(
    pattern="stats$",
)
async def stats(
    event: NewMessage.Event,
):
    ok = await event.eor("`Collecting stats...`")
    start_time = time.time()
    private_chats = 0
    bots = 0
    groups = 0
    broadcast_channels = 0
    admin_in_groups = 0
    creator_in_groups = 0
    admin_in_broadcast_channels = 0
    creator_in_channels = 0
    unread_mentions = 0
    unread = 0
    dialog: Dialog
    async for dialog in event.client.iter_dialogs():
        entity = dialog.entity
        if isinstance(entity, Channel) and entity.broadcast:
            broadcast_channels += 1
            if entity.creator or entity.admin_rights:
                admin_in_broadcast_channels += 1
            if entity.creator:
                creator_in_channels += 1

        elif (isinstance(entity, Channel) and entity.megagroup) or isinstance(
            entity, Chat
        ):
            groups += 1
            if entity.creator or entity.admin_rights:
                admin_in_groups += 1
            if entity.creator:
                creator_in_groups += 1

        elif isinstance(entity, User):
            private_chats += 1
            if entity.bot:
                bots += 1

        unread_mentions += dialog.unread_mentions_count
        unread += dialog.unread_count
    stop_time = time.time() - start_time
    try:
        ct = (await event.client(GetBlockedRequest(1, 0))).count
    except AttributeError:
        ct = 0
    try:
        sp = await event.client(GetAllStickersRequest(0))
        sp_count = len(sp.sets)
    except BaseException:
        sp_count = 0
    full_name = inline_mention(event.client.me)
    response = f"🔸 **Stats for {full_name}** \n\n"
    response += f"**Private Chats:** {private_chats} \n"
    response += f"**  •• **`Users: {private_chats - bots}` \n"
    response += f"**  •• **`Bots: {bots}` \n"
    response += f"**Groups:** {groups} \n"
    response += f"**Channels:** {broadcast_channels} \n"
    response += f"**Admin in Groups:** {admin_in_groups} \n"
    response += f"**  •• **`Creator: {creator_in_groups}` \n"
    response += f"**  •• **`Admin Rights: {admin_in_groups - creator_in_groups}` \n"
    response += f"**Admin in Channels:** {admin_in_broadcast_channels} \n"
    response += f"**  •• **`Creator: {creator_in_channels}` \n"
    response += f"**  •• **`Admin Rights: {admin_in_broadcast_channels - creator_in_channels}` \n"
    response += f"**Unread:** {unread} \n"
    response += f"**Unread Mentions:** {unread_mentions} \n"
    response += f"**Blocked Users:** {ct}\n"
    response += f"**Total Stickers Pack Installed :** `{sp_count}`\n\n"
    response += f"**__It Took:__** {stop_time:.02f}s \n"
    await ok.edit(response)


@ultroid_cmd(pattern="paste( (.*)|$)", manager=True, allow_all=True)
async def _(event):
    try:
        input_str = event.text.split(maxsplit=1)[1]
    except IndexError:
        input_str = None

    xx = await event.eor("` 《 Pasting... 》 `")
    downloaded_file_name, ext = None, ".txt"
    if input_str:
        message = input_str
    elif event.reply_to_msg_id:
        previous_message = await event.get_reply_message()
        if previous_message.media:
            downloaded_file_name = await event.client.download_media(
                previous_message,
                "resources/downloads",
            )
            ext = getattr(previous_message.file, "ext", "txt")
            message = await asyncread(downloaded_file_name)
            osremove(downloaded_file_name)
        else:
            message = previous_message.message
    else:
        message = None

    if not message:
        return await xx.eor(
            "`Reply to a Message/Document or Give me Some Text !`", time=5
        )

    done, data = await get_paste(message, extension=ext[1:] if ext else "txt")
    if not done and data.get("error"):
        return await xx.eor(f"Something Went Wrong! `{data['error']}`")
    link = data["link"]
    raw = data["raw"]
    reply_text = f"• **Pasted to SpaceBin :** [SpaceBin]({data['link']})\n• **Raw Url :** : [Raw]({data['raw']})"
    try:
        if event.client._bot:
            return await xx.eor(reply_text)
        ok = await event.client.inline_query(asst.me.username, f"pasta-{data['link']}")
        await ok[0].click(event.chat_id, reply_to=event.reply_to_msg_id, hide_via=True)
        await xx.delete()
    except BaseException as e:
        LOGS.exception(e)
        await xx.edit(reply_text)


@ultroid_cmd(
    pattern="info( (.*)|$)",
    manager=True,
)
async def _(event):
    if match := event.pattern_match.group(1).strip():
        try:
            user = await event.client.parse_id(match)
        except Exception as er:
            return await event.eor(str(er))
    elif event.is_reply:
        rpl = await event.get_reply_message()
        user = rpl.sender_id
    else:
        user = event.chat_id
    xx = await event.eor(get_string("com_1"))
    try:
        _ = await event.client.get_entity(user)
    except Exception as er:
        return await xx.edit(f"**ERROR :** {er}")
    if not isinstance(_, User):
        try:
            peer = get_peer_id(_)
            photo, capt = await get_chat_info(_, event)
            if is_gbanned(peer):
                capt += "\n•<b> Is Gbanned:</b> <code>True</code>"
            if not photo:
                return await xx.eor(capt, parse_mode="html")
            await event.client.send_message(
                event.chat_id, capt[:1024], file=photo, parse_mode="html"
            )
            await xx.delete()
        except Exception as er:
            await event.eor("**ERROR ON CHATINFO**\n" + str(er))
        return
    try:
        full_user = (await event.client(GetFullUserRequest(user))).full_user
    except Exception as er:
        return await xx.edit(f"ERROR : {er}")
    user = _
    user_photos = (
        await event.client.get_profile_photos(user.id, limit=0)
    ).total or "NaN"
    user_id = user.id
    first_name = user.first_name
    if first_name:
        first_name = html.escape(first_name)
        first_name = first_name.replace(r"\u2060", "")
    last_name = user.last_name
    last_name = (
        last_name.replace(r"\u2060", "") if last_name else ("Last Name not found")
    )
    user_bio = full_user.about
    if user_bio is not None:
        user_bio = html.escape(full_user.about)
    common_chats = full_user.common_chats_count
    if user.photo:
        dc_id = user.photo.dc_id
    else:
        dc_id = "Need a Profile Picture to check this"
    caption = """<b>Exᴛʀᴀᴄᴛᴇᴅ Dᴀᴛᴀ Fʀᴏᴍ Tᴇʟᴇɢʀᴀᴍ's Dᴀᴛᴀʙᴀsᴇ<b>
<b>••Tᴇʟᴇɢʀᴀᴍ ID</b>: <code>{}</code>
<b>••Pᴇʀᴍᴀɴᴇɴᴛ Lɪɴᴋ</b>: <a href='tg://user?id={}'>Click Here</a>
<b>••Fɪʀsᴛ Nᴀᴍᴇ</b>: <code>{}</code>
<b>••Sᴇᴄᴏɴᴅ Nᴀᴍᴇ</b>: <code>{}</code>
<b>••Bɪᴏ</b>: <code>{}</code>
<b>••Dᴄ ID</b>: <code>{}</code>
<b>••Nᴏ. Oғ PғPs</b> : <code>{}</code>
<b>••Is Rᴇsᴛʀɪᴄᴛᴇᴅ</b>: <code>{}</code>
<b>••Vᴇʀɪғɪᴇᴅ</b>: <code>{}</code>
<b>••Is Pʀᴇᴍɪᴜᴍ</b>: <code>{}</code>
<b>••Is A Bᴏᴛ</b>: <code>{}</code>
<b>••Gʀᴏᴜᴘs Iɴ Cᴏᴍᴍᴏɴ</b>: <code>{}</code>
""".format(
        user_id,
        user_id,
        first_name,
        last_name,
        user_bio,
        dc_id,
        user_photos,
        user.restricted,
        user.verified,
        user.premium,
        user.bot,
        common_chats,
    )
    if chk := is_gbanned(user_id):
        caption += f"""<b>••Gʟᴏʙᴀʟʟʏ Bᴀɴɴᴇᴅ</b>: <code>True</code>
<b>••Rᴇᴀsᴏɴ</b>: <code>{chk}</code>"""
    await event.client.send_message(
        event.chat_id,
        caption,
        reply_to=event.reply_to_msg_id,
        parse_mode="HTML",
        file=full_user.profile_photo,
        force_document=False,
        silent=True,
    )
    await xx.delete()


@ultroid_cmd(
    pattern="invite( (.*)|$)",
    groups_only=True,
)
async def _(ult):
    xx = await ult.eor(get_string("com_1"))
    to_add_users = ult.pattern_match.group(1).strip()
    if not ult.is_channel and ult.is_group:
        for user_id in to_add_users.split(" "):
            try:
                await ult.client(
                    AddChatUserRequest(
                        chat_id=ult.chat_id,
                        user_id=await ult.client.parse_id(user_id),
                        fwd_limit=1000000,
                    ),
                )
                await xx.edit(f"Successfully invited `{user_id}` to `{ult.chat_id}`")
            except Exception as e:
                await xx.edit(str(e))
    else:
        for user_id in to_add_users.split(" "):
            try:
                await ult.client(
                    InviteToChannelRequest(
                        channel=ult.chat_id,
                        users=[await ult.client.parse_id(user_id)],
                    ),
                )
                await xx.edit(f"Successfully invited `{user_id}` to `{ult.chat_id}`")
            except UserBotError:
                await xx.edit(
                    f"Bots can only be added as Admins in Channel.\nBetter Use `{HNDLR}promote {user_id}`"
                )
            except Exception as e:
                await xx.edit(str(e))


@ultroid_cmd(
    pattern="rmbg($| (.*))",
)
async def abs_rmbg(event):
    RMBG_API = udB.get_key("RMBG_API")
    if not Image:
        return await event.eor(f"`Install 'Pillow' module to use this..`")
    if not RMBG_API:
        return await event.eor(
            "Get your API key from [here](https://www.remove.bg/) for this plugin to work.",
        )

    match = event.pattern_match.group(1).strip()
    reply = await event.get_reply_message()
    if match and os.path.exists(match):
        dl = match
    elif reply and reply.media:
        if reply.document and reply.document.thumbs:
            dl = await reply.download_media(thumb=-1)
        else:
            dl = await reply.download_media()
    else:
        return await eod(
            event, f"Use `{HNDLR}rmbg` as reply to a pic to remove its background."
        )
    if not (dl and dl.endswith(("webp", "jpg", "png", "jpeg"))):
        osremove(dl)
        return await event.eor(get_string("com_4"))
    if dl.endswith("webp"):
        file = f"{dl}.png"
        Image.open(dl).save(file)
        osremove(dl)
        dl = file
    xx = await event.eor("`Sending to remove.bg`")
    dn, out = await ReTrieveFile(dl)
    osremove(dl)
    if not dn:
        dr = out["errors"][0]
        de = dr.get("detail", "")
        return await xx.edit(
            f"**ERROR ~** `{dr['title']}`,\n`{de}`",
        )
    zz = Image.open(out)
    if zz.mode != "RGB":
        zz.convert("RGB")
    wbn = check_filename("ult-rmbg.webp")
    zz.save(wbn, "webp")
    await event.client.send_file(
        event.chat_id,
        out,
        force_document=True,
        reply_to=reply,
    )
    await event.client.send_file(event.chat_id, wbn, reply_to=reply)
    osremove(out, wbn)
    await xx.delete()


@ultroid_cmd(
    pattern="(telegraph|catbox)( (.*)|$)",
)
async def telegraphcmd(event):
    xx = await event.eor(get_string("com_1"))
    match = event.pattern_match.group(3) or "Ultroid"
    reply = await event.get_reply_message()
    if not reply:
        return await xx.eor("`Reply to Message.`")

    if not reply.media and reply.message:
        content = reply.message
    else:
        getit = await reply.download_media()
        dar = mediainfo(reply.media)
        if dar == "sticker":
            file = f"{getit}.png"
            if not Image:
                osremove(getit)
                return await event.eor(f"`Install 'Pillow' module to use this..`")

            Image.open(getit).save(file)
            osremove(getit)
            getit = file
        elif dar.endswith("animated"):
            file = f"{getit}.gif"
            await bash(f"lottie_convert.py {shquote(getit)} {file}")
            osremove(getit)
            getit = file

        if "document" not in dar:
            try:
                nn = await Catbox(getit)
                amsg = f"Uploaded to [CatBox]({nn}) !"
            except Exception as e:
                amsg = f"Error: {e}"
            osremove(getit)
            return await xx.eor(amsg)

        content = pathlib.Path(getit).read_text()
        osremove(getit)

    makeit = await TelegraphClient.create_page(title=match, content=[content])
    await xx.eor(f"Pasted to Telegraph : [Telegraph]({makeit})", link_preview=False)


@ultroid_cmd(pattern="json( (.*)|$)")
async def _(event):
    reply_to_id = None
    match = event.pattern_match.group(1).strip()
    if event.reply_to_msg_id:
        msg = await event.get_reply_message()
        reply_to_id = event.reply_to_msg_id
    else:
        msg = event
        reply_to_id = event.message.id
    if match and hasattr(msg, match.split()[0]):
        msg = getattr(msg, match.split()[0])
        try:
            if hasattr(msg, "to_json"):
                msg = msg.to_json(ensure_ascii=False, indent=1)
            elif hasattr(msg, "to_dict"):
                msg = json_parser(msg.to_dict(), indent=1)
            else:
                msg = TLObject.stringify(msg)
        except Exception:
            pass
        msg = str(msg)
    else:
        msg = json_parser(msg.to_json(), indent=1)
    if "-t" in match:
        try:
            data = json_parser(msg)
            msg = json_parser(
                {key: data[key] for key in data.keys() if data[key]}, indent=1
            )
        except Exception:
            pass
    if len(msg) > 4096:
        with io.BytesIO(str.encode(msg)) as out_file:
            out_file.name = "json-ult.txt"
            await event.client.send_file(
                event.chat_id,
                out_file,
                reply_to=reply_to_id,
            )
            await event.delete()
    else:
        msg = (
            f"""<pre><code class="language-json">{msg}</code></pre>"""
            if msg
            else "<code>empty</code>"
        )
        await event.eor(msg, parse_mode="html")


@ultroid_cmd(pattern="suggest( (.*)|$)", manager=True)
async def sugg(event):
    sll = event.text.split(maxsplit=1)
    try:
        text = sll[1]
    except IndexError:
        text = None
    if not (event.is_reply or text):
        return await eod(
            event,
            "`Please reply to a message to make a suggestion poll!`",
        )
    if event.is_reply and not text:
        reply = await event.get_reply_message()
        if reply.text and len(reply.text) < 35:
            text = reply.text
        else:
            text = "Do you Agree to Replied Suggestion ?"
    reply_to = event.reply_to_msg_id if event.is_reply else event.id
    try:
        await event.client.send_file(
            event.chat_id,
            file=InputMediaPoll(
                poll=Poll(
                    id=12345,
                    question=TextWithEntities(text, entities=[]),
                    answers=[
                        PollAnswer(TextWithEntities("Yes", entities=[]), b"1"),
                        PollAnswer(TextWithEntities("No", entities=[]), b"2"),
                    ],
                ),
            ),
            reply_to=reply_to,
        )
    except Exception as e:
        return await eod(event, f"`Oops, you can't send polls here!\n\n{e}`")
    await event.delete()


@ultroid_cmd(pattern="ipinfo( (.*)|$)")
async def ipinfo(event):
    ip = event.text.split()
    ipaddr = ""
    try:
        ipaddr = f"/{ip[1]}"
    except IndexError:
        ipaddr = ""
    det = await async_searcher(f"https://ipinfo.io{ipaddr}/geo", re_json=True)
    try:
        ip = det["ip"]
        city = det["city"]
        region = det["region"]
        country = det["country"]
        cord = det["loc"]
        try:
            zipc = det["postal"]
        except KeyError:
            zipc = "None"
        tz = det["timezone"]
        await eor(
            event,
            """
**IP Details Fetched.**

**IP:** `{}`
**City:** `{}`
**Region:** `{}`
**Country:** `{}`
**Co-ordinates:** `{}`
**Postal Code:** `{}`
**Time Zone:** `{}`
""".format(
                ip,
                city,
                region,
                country,
                cord,
                zipc,
                tz,
            ),
        )
    except BaseException:
        err = det["error"]["title"]
        msg = det["error"]["message"]
        await event.eor(f"ERROR:\n{err}\n{msg}", time=5)


@ultroid_cmd(
    pattern="cpy$",
)
async def copp(event):
    msg = await event.get_reply_message()
    if not msg:
        return await event.eor(f"Use `{HNDLR}cpy` as reply to a message!", time=5)
    _copied_msg["CLIPBOARD"] = msg
    await event.eor(f"Copied. Use `{HNDLR}pst` to paste!", time=10)


@asst_cmd(pattern="pst$")
async def pepsodent(event):
    await toothpaste(event)


@ultroid_cmd(
    pattern="pst$",
)
async def colgate(event):
    await toothpaste(event)


async def toothpaste(event):
    try:
        await event.respond(_copied_msg["CLIPBOARD"])
    except KeyError:
        return await eod(
            event,
            f"Nothing was copied! Use `{HNDLR}cpy` as reply to a message first!",
        )
    except Exception as ex:
        return await event.eor(str(ex), time=5)
    await event.delete()


@ultroid_cmd(pattern="thumb$")
async def thumb_dl(event):
    reply = await event.get_reply_message()
    if not (reply and reply.media and mediainfo(reply.media) not in ("web", "")):
        return await eod(event, get_string("th_1"), time=5)

    myy = await event.eor(get_string("com_1"))
    try:
        m = await reply.download_media(thumb=-1, file=bytes)
    except IndexError:
        return await eod(myy, get_string("th_2"))

    m = io.BytesIO(m)
    m.name = "thumb.jpg"
    await asyncio.gather(
        reply.reply(f"`Generated Thumbnail`", file=m),
        myy.delete(),
        return_exceptions=True,
    )


@ultroid_cmd(pattern="getmsg( ?(.*)|$)")
async def get_restriced_msg(event):
    match = event.pattern_match.group(1).strip()
    if not match:
        await event.eor("`Please provide a link!`", time=5)
        return
    xx = await event.eor(get_string("com_1"))
    chat, msg = get_chat_and_msgid(match)
    if not (chat and msg):
        return await event.eor(
            f"{get_string('gms_1')}!\nEg: `https://t.me/TeamUltroid/3 or `https://t.me/c/1313492028/3`"
        )
    try:
        message = await event.client.get_messages(chat, ids=msg)
    except BaseException as er:
        return await event.eor(f"**ERROR**\n`{er}`")
    try:
        await event.client.send_message(event.chat_id, message)
        await xx.try_delete()
        return
    except ChatForwardsRestrictedError:
        pass
    if message.media:
        thumb, attributes = None, []
        media, _ = await tg_downloader(media=message, event=xx, show_progress=True)
        if doc := message.document:
            attributes = doc.attributes
            if doc.thumbs:
                thumb = await message.download_media(thumb=-1)

        await xx.edit("`Uploading...`")
        uploaded, _ = await event.client.fast_uploader(
            media, event=xx, show_progress=True, to_delete=True
        )
        typ = not bool(message.video)
        await event.reply(
            message.text,
            file=uploaded,
            supports_streaming=typ,
            force_document=typ,
            thumb=thumb,
            attributes=attributes,
        )
        await xx.delete()
        if thumb:
            osremove(thumb)

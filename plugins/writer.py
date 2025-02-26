# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://www.github.com/TeamUltroid/Ultroid/blob/main/LICENSE/>.

"""
✘ Commands Available -

• `{i}gethtml <url>`
   Get HTML elements of any Site.

• `{i}write <text or reply to text>`
   It will write on a paper.

• `{i}image <text or reply to html or any doc file>`
   Write a image from html or any text.
"""

import os

from htmlwebshot import WebShot
from PIL import Image, ImageDraw, ImageFont

from . import (
    async_searcher,
    asyncwrite,
    check_filename,
    get_string,
    LOGS,
    osremove,
    text_set,
    ultroid_cmd,
)


@ultroid_cmd(pattern="gethtml( (.*)|$)")
async def ghtml(e):
    if txt := e.pattern_match.group(2):
        link = e.text.split(maxsplit=1)[1]
    else:
        return await e.eor("`Give any URL to generate html elements.`", time=5)
    m = await e.eor(get_string("com_1"))
    k = await async_searcher(link)
    file = check_filename("file.html")
    await asyncwrite(file, k, mode="w+")
    await m.respond(f"`{link}`", file=file)
    osremove(file)
    await m.delete()


@ultroid_cmd(pattern="image( (.*)|$)")
async def f2i(e):
    txt = e.pattern_match.group(1).strip()
    html = None
    if txt:
        html = e.text.split(maxsplit=1)[1]
    elif e.reply_to:
        r = await e.get_reply_message()
        if r.media:
            html = await e.client.download_media(r.media)
        elif r.text:
            html = r.text
    if not html:
        return await e.eor("`Either reply to any file or give any text`", time=5)
    m = await e.eor(get_string("com_1"))
    html = html.replace("\n", "<br>")
    shot = WebShot(quality=90)
    css = "body {background: white;} p {color: red;}"
    pic = await shot.create_pic_async(html=html, css=css)
    await e.client.send_file(e.chat_id, pic, reply_to=e.reply_to_msg_id)
    osremove(pic, html)
    await m.delete()


@ultroid_cmd(pattern=r"write( ([\s\S]*)|$)")
async def writer(e):
    if e.reply_to:
        reply = await e.get_reply_message()
        text = reply.message
    elif inpt := e.pattern_match.group(2):
        text = inpt
    else:
        return await e.eor(get_string("writer_1"), time=5)
    k = await e.eor(get_string("com_1"))
    img = Image.open("resources/extras/template.jpg")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("resources/fonts/assfont.ttf", size=30)
    x, y = 150, 140
    lines = text_set(text)
    bbox = font.getbbox("hg")
    line_height = bbox[3] - bbox[1]
    for line in lines:
        draw.text((x, y), line, fill=(1, 22, 55), font=font)
        y = y + line_height - 5
    file = check_filename("ult-writer.jpg")
    img.save(file)
    await e.client.send_file(e.chat_id, file, reply_to=e.reply_to_msg_id)
    osremove(file)
    await k.delete()

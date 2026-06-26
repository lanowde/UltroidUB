# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

__all__ = ("GDriveManager",)


import json
import os
import time
from functools import wraps
from mimetypes import guess_type
from urllib.parse import parse_qs, urlencode

import aiofiles
from aiohttp import ClientSession

from pyUltroid import udB
from pyUltroid.custom.commons import (
    check_filename,
    humanbytes,
    random_string,
    time_formatter,
)


class GDriveManager:
    __slots__ = (
        "base_url",
        "client_id",
        "client_secret",
        "folder_id",
        "key_suffix",
        "scope",
        "creds",
    )

    def __init__(self, key_suffix=None):
        self.key_suffix = key_suffix or ""
        self.base_url = "https://www.googleapis.com/drive/v3"
        self.client_id = (
            udB.get_key(self._fix_keys("GDRIVE_CLIENT_ID"))
            or "458306970678-jhfbv6o5sf1ar63o1ohp4c0grblp8qba.apps.googleusercontent.com"
        )
        self.client_secret = (
            udB.get_key(self._fix_keys("GDRIVE_CLIENT_SECRET"))
            or "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"
        )
        self.folder_id = udB.get_key(self._fix_keys("GDRIVE_FOLDER_ID")) or "root"
        self.scope = "https://www.googleapis.com/auth/drive"
        self.creds = udB.get_key(self._fix_keys("GDRIVE_AUTH_TOKEN")) or {}

    # hack for accessing multiple gdrive
    def _fix_keys(self, key):
        return key + "_" + (self.key_suffix or "")

    @staticmethod
    def extract_drive_id(link):
        if "?id=" in link:
            # https://drive.google.com/uc?id=1c7dE6hiYnTKlyBnWV4HrdGMkAhun_FZY&export=download
            if file_id := parse_qs(link).get("id"):
                return file_id[0]
        elif "file/d/" in link:
            # https://drive.google.com/file/d/1mFKVR1_eNOf279TD_KrAvkHOKPiOcW2Y/view?usp=drive_link
            spl = link.lsplit("file/d/", maxsplit=1)[1]
            return spl.lsplit("/", maxsplit=1)[0] if "/" in spl else spl

    @staticmethod
    def upload_chunk_size(file_size):
        if file_size < 8 * 1024 * 1024:
            return 128 * 1024  # 128KB blocks
        elif file_size < 128 * 1024 * 1024:
            return 2 * 1024 * 1024  # 2MB blocks
        else:
            return 32 * 1024 * 1024  # 32MB blocks

    @staticmethod
    def download_chunk_size(file_size):
        if file_size == 1:  # server side mismatch
            return 32 * 1024 * 1024  # use default blocks
        elif file_size < 200 * 1024 * 1024:
            return 8 * 1024 * 1024  # 8MB blocks
        else:
            return 32 * 1024 * 1024  # 32MB blocks

    def check_access_token(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if time.time() > self.creds.get("expires_in"):
                await self.refresh_access_token()
            return await func(*args, **kwargs)

        return wrapper

    def get_oauth2_url(self):
        # this url returns one-time-usabe codes, like: 4/1AdkVLVne2..
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(
            {
                "client_id": self.client_id,
                "redirect_uri": "http://localhost",
                "response_type": "code",
                "scope": self.scope,
                "access_type": "offline",
                "prompt": "consent",
            }
        )

    async def get_access_token(self, code=None) -> dict | str:
        if not code:
            return self.get_oauth2_url()
        if code.startswith("http://localhost"):
            code = parse_qs(code.split("?")[1]).get("code")[0]
        url = "https://oauth2.googleapis.com/token"
        async with ClientSession() as client:
            resp = await client.post(
                url,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uri": "http://localhost",
                    "grant_type": "authorization_code",
                    "code": code,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            self.creds = await resp.json()
        self.creds["expires_in"] = time.time() + 3590
        udB.set_key(self._fix_keys("GDRIVE_AUTH_TOKEN"), self.creds)
        return True

    async def refresh_access_token(self) -> None:
        async with ClientSession() as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": self.creds.get("refresh_token"),
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            self.creds["access_token"] = (await resp.json())["access_token"]
        self.creds["expires_in"] = time.time() + 3590
        udB.set_key(self._fix_keys("GDRIVE_AUTH_TOKEN"), self.creds)

    @check_access_token
    async def upload_file(self, event, path: str, folder_id: str | bool = None):
        mime_type = guess_type(path)[0] or "application/octet-stream"
        # upload with progress bar
        filesize = os.path.getsize(path)
        filename = os.path.basename(path)
        chunksize = GDriveManager.upload_chunk_size(filesize)

        # 1. Retrieve session for resumable upload.
        headers = {
            "Authorization": "Bearer " + self.creds.get("access_token"),
            "Content-Type": "application/json",
        }
        params = {
            "name": filename,
            "mimeType": mime_type,
            "fields": "id, name, webContentLink",
            "parents": [folder_id] if folder_id else [self.folder_id],
        }
        async with ClientSession() as client:
            r = await client.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
                headers=headers,
                data=json.dumps(params),
                params={"fields": "id, name, webContentLink"},
            )
            if r.status == 401:
                await self.refresh_access_token()
                return await self.upload_file(event, path, folder_id)
            elif r.status == 403:
                # upload to root and move
                return await self.upload_file(event, path, "root")

            upload_url = r.headers.get("Location")

        uploaded = 0
        response = None
        start = time.time()
        last_edit_time = start - 7
        async with aiofiles.open(path, mode="rb") as f:
            while filesize != uploaded:
                chunk_data = await f.read(chunksize)
                headers = {
                    "Content-Length": str(len(chunk_data)),
                    "Content-Range": f"bytes {uploaded}-{uploaded + len(chunk_data) - 1}/{filesize}",
                }
                uploaded += len(chunk_data)
                async with ClientSession() as client:
                    response = await client.put(
                        upload_url,
                        data=chunk_data,
                        headers=headers,
                    )

                now = time.time()
                if now - last_edit_time < 6:
                    continue
                diff = now - start
                percentage = round((uploaded / filesize) * 100, 2)
                speed = round(uploaded / diff, 2)
                eta = round((filesize - uploaded) / speed, 2) * 1000
                current_txt = (
                    f"Uploading `{filename}` to **GDrive**...\n\n"
                    + f"**Status:**  `{humanbytes(uploaded)}/{humanbytes(filesize)}` » `{percentage}%`\n"
                    + f"**Speed:**  `{humanbytes(speed)}/s`\n"
                    + f"**ETA:**  `{time_formatter(eta)}`"
                )
                await event.edit(current_txt)
                last_edit_time = now
            return await response.json()

    @check_access_token
    async def download_file(self, event, file_id: str):
        fileId = GDriveManager.extract_drive_id(file_id)
        headers = {
            "Authorization": "Bearer " + self.creds.get("access_token"),
            "Content-Type": "application/json",
        }
        params = {
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "fields": "id, name, mimeType, size",
            "parents": [self.folder_id],
        }
        async with ClientSession() as client:
            r = await client.get(
                self.base_url + f"/files/{fileId}",
                headers=headers,
                params=params,
            )
            if r.status != 200:
                try:
                    js = await r.json()
                except Exception:
                    js = await r.text()
                return False, js

            resp = await r.json()

        filename = resp.get("name", random_string(12))
        filename = check_filename(f"resources/downloads/{filename}")
        filesize = int(resp.get("size", 1))
        downloaded = 0
        start = time.time()
        last_edit_time = start - 7
        chunksize = GDriveManager.download_chunk_size(filesize)
        async with aiofiles.open(filename, "wb") as f:
            async with ClientSession() as client:
                resp1 = await client.get(
                    self.base_url + f"/files/{fileId}",
                    headers=headers,
                    params={"alt": "media", **params},
                    timeout=None,
                )
                async for chunk in resp1.content.iter_chunked(chunksize):
                    downloaded += await f.write(chunk)
                    now = time.time()
                    if now - last_edit_time < 6.5:
                        continue
                    diff = now - start
                    percentage = round((downloaded / filesize) * 100, 2)
                    speed = round(downloaded / diff, 2)
                    eta = round((filesize - downloaded) / speed, 2) * 1000
                    current_txt = (
                        f"Downloading `{filename}` from GDrive...\n\n"
                        + f"**Status:**  `{humanbytes(downloaded)}/{humanbytes(filesize)}` » `{percentage}%`\n"
                        + f"**Speed:**  `{humanbytes(speed)}/s`\n"
                        + f"**ETA:**  `{time_formatter(eta)}`"
                    )
                    await event.edit(current_txt)
                    last_edit_time = now

        return True, filename

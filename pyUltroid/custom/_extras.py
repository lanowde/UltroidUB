# additional functions that must be seperate from commons..

__all__ = (
    "aiohttp",
    "cpu_bound",
    "run_async",
    "async_searcher",
    "FixedSizeDict",
)

import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import UserDict
from functools import partial, wraps

from pyUltroid.exceptions import DependencyMissingError
from ._loop import loop

try:
    import requests
except ImportError:
    requests = None

try:
    import aiohttp
except ImportError:
    aiohttp = None


"""
# this shit doesn't works and causes weird errors

try:
    import aiodns
except ImportError:
    aiodns = None

# dns for async_searcher
_kwargs = {"ttl_dns_cache": 120, "loop": loop}
if aiohttp and aiodns:
    resolver = aiohttp.resolver.AsyncResolver(
        nameservers=[
            "1.1.1.1",
            "1.0.0.1",
            "2606:4700:4700::1111",
            "2606:4700:4700::1001",
            "8.8.8.8",
        ]
    )
    _kwargs["resolver"] = resolver

connector = aiohttp.TCPConnector(**_kwargs)
"""


_workers = __import__("multiprocessing").cpu_count()


# source: fns/helper.py
# preferred for I/O bound task.
def run_async(function):
    @wraps(function)
    async def wrapper(*args, **kwargs):
        return await loop.run_in_executor(
            ThreadPoolExecutor(max_workers=_workers * 2),
            partial(function, *args, **kwargs),
        )

    return wrapper


# preferred for cpu bound tasks.
def cpu_bound(function):
    @wraps(function)
    async def wrapper(*args, **kwargs):
        with ProcessPoolExecutor(max_workers=_workers) as pool:
            output = await loop.run_in_executor(
                pool,
                partial(function, *args, **kwargs),
            )
        return output

    return wrapper


class FixedSizeDict(UserDict):
    """dict with maxsize"""

    __slots__ = ("maxsize", "data")

    def __init__(self, *args, **kwargs):
        self.maxsize = kwargs.pop("maxsize", 32)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        self.data[key] = value
        if len(self.data) > self.maxsize:
            self.data.pop(next(iter(self.data)), None)


# source: fns/helper.py
# Async Searcher -> @buddhhu
async def async_searcher(
    url: str,
    post: bool = False,
    method: str = "GET",
    headers: dict = None,
    evaluate: callable = None,
    object: bool = False,
    re_json: bool = False,
    re_content: bool = False,
    timeout: int | bool = 60,
    *args,
    **kwargs,
):
    method = "POST" if post else method.upper()

    if False:  # no aiohttp.
        if timeout:
            timeout = aiohttp.ClientTimeout(total=int(timeout))
        async with aiohttp.ClientSession(headers=headers) as client:
            data = await client.request(method, url, *args, timeout=timeout, **kwargs)
            if evaluate:
                from telethon.helpers import _maybe_await

                return await _maybe_await(evaluate(data))
            elif re_json:
                return await data.json()
            elif re_content:
                return await data.read()
            elif object:
                return data
            else:
                return await data.text()

    elif requests:
        if "ssl" in kwargs:
            kwargs["verify"] = kwargs.pop("ssl", None)

        @run_async
        def sync_request():
            data = requests.request(
                method,
                url,
                *args,
                headers=headers,
                timeout=timeout,
                **kwargs,
            )
            if re_json:
                return data.json()
            elif re_content:
                return data.content
            elif evaluate or object:
                return data
            else:
                return data.text

        data = await sync_request()
        if evaluate:
            from telethon.helpers import _maybe_await

            return await _maybe_await(evaluate(data))
        return data

    else:
        raise DependencyMissingError("Install 'aiohttp' or 'requests' to use this.")

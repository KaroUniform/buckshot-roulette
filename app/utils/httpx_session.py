"""aiogram session backed by httpx with first-class HTTPS-proxy support.

aiohttp (aiogram's default transport) handles TLS-tunnel proxies
inconsistently across versions — connections either silently hang or
give cryptic ``ClientProxyConnectionError``s. httpx supports the same
`https://user:pass@host:port` URL scheme and just works. This module
ports the session wrapper from the neighbouring rambirds bot (same
operator's Telegram stack) so both bots share the same transport when
api.telegram.org isn't directly reachable.

Used only when the ``HTTPS_PROXY`` environment variable is set; see
``main.py`` for wiring.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

import httpx

from aiogram.exceptions import TelegramNetworkError
from aiogram.methods.base import TelegramType
from aiogram.client.session.base import BaseSession

if TYPE_CHECKING:
    from aiogram.client.bot import Bot
    from aiogram.methods import TelegramMethod
    from aiogram.types import InputFile


class HttpxSession(BaseSession):
    """``BaseSession`` backed by ``httpx.AsyncClient`` with proxy support."""

    def __init__(self, proxy: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._proxy = proxy
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(proxy=self._proxy)
        return self._client

    async def _collect_file_bytes(self, input_file: "InputFile", bot: "Bot") -> bytes:
        chunks = []
        async for chunk in input_file.read(bot):
            chunks.append(chunk)
        return b"".join(chunks)

    async def make_request(
        self,
        bot: "Bot",
        method: "TelegramMethod[TelegramType]",
        timeout: int | None = None,
    ) -> TelegramType:
        client = await self._get_client()
        url = self.api.api_url(token=bot.token, method=method.__api_method__)

        data: dict[str, Any] = {}
        files_dict: dict[str, "InputFile"] = {}

        for key, value in method.model_dump(warnings=False).items():
            value = self.prepare_value(value, bot=bot, files=files_dict)
            if not value:
                continue
            data[key] = value

        httpx_files: list[tuple[str, tuple[str, bytes]]] = []
        for key, input_file in files_dict.items():
            file_bytes = await self._collect_file_bytes(input_file, bot)
            httpx_files.append((key, (input_file.filename or key, file_bytes)))

        req_timeout = self.timeout if timeout is None else timeout

        try:
            resp = await client.post(
                url,
                data=data,
                files=httpx_files or None,
                timeout=req_timeout,
            )
        except httpx.TimeoutException as e:
            raise TelegramNetworkError(method=method, message="Request timeout error") from e
        except httpx.HTTPError as e:
            raise TelegramNetworkError(method=method, message=f"{type(e).__name__}: {e}") from e

        response = self.check_response(
            bot=bot,
            method=method,
            status_code=resp.status_code,
            content=resp.text,
        )
        return cast(TelegramType, response.result)

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        client = await self._get_client()
        async with client.stream(
            "GET", url, headers=headers or {}, timeout=timeout,
        ) as resp:
            if raise_for_status:
                resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size):
                yield chunk

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

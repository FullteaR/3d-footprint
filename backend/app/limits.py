"""A ceiling on the request body, applied before anything parses it.

Uploads arrive as multipart, and Starlette spools each part into a temporary
file as it parses — so a size check written inside the route runs *after* the
bytes have already been written to disk. A limit that only lives there is a
validation message, not a defence: it cannot stop a request from filling the
disk, because the request is already on it.

This sits in front of the application instead. A declared Content-Length over
the ceiling is answered on the spot, without reading a byte of the body, which
is the case every ordinary client falls into. A body that arrives without one
(chunked) is counted as it streams and cut off at the same ceiling.
"""
from __future__ import annotations

import json


def _declared_length(scope) -> int | None:
    """The request's Content-Length, or None if absent or malformed."""
    for key, value in scope.get("headers", []):
        if key == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _reply_too_large(send, limit: int) -> None:
    """A 413 shaped like the app's other errors, so the UI shows the reason."""
    body = json.dumps(
        {"detail": f"リクエストが大きすぎます（上限 {limit // (1024 * 1024)} MB）"},
        ensure_ascii=False,
    ).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [(b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode())],
    })
    await send({"type": "http.response.body", "body": body})


class LimitRequestBody:
    """ASGI middleware refusing any request body past `limit` bytes."""

    def __init__(self, app, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.limit:
            await _reply_too_large(send, self.limit)
            return

        seen = 0
        over = False

        async def counted_receive():
            nonlocal seen, over
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    over = True
                    # Hanging up is what stops the multipart parser mid-stream.
                    # Whatever the app makes of that is dropped below.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message) -> None:
            if not over:
                await send(message)

        try:
            await self.app(scope, counted_receive, guarded_send)
        except Exception:
            # A disconnect we invented ourselves, surfacing as whatever the
            # parser raises. Anything else is a real failure and belongs to
            # the handler above.
            if not over:
                raise
        if over:
            await _reply_too_large(send, self.limit)

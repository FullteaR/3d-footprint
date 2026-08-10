"""FastAPI app: serves the /api/* routes and the built frontend (single container)."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router as api_router
from .config import CORS_ORIGINS, MAX_REQUEST_BYTES, STATIC_DIR
from .limits import LimitRequestBody

app = FastAPI(title="3d-footprint", version="0.1.0")

# Order matters, and it is the reverse of the order added: the last one wrapped
# is the outermost. CORS goes on last so its headers reach the body-size 413
# too — without them a browser reports the refusal as a CORS failure and the
# reason never gets to the user.
app.add_middleware(LimitRequestBody, limit=MAX_REQUEST_BYTES)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def _readable_validation_error(request: Request, exc: RequestValidationError):
    """Say which field was out of range, in the shape the UI already reads.

    FastAPI answers a bad parameter with `detail` as a *list* of dicts, while
    every other error here puts a sentence there — the front end shows it
    straight, so the reader would get "[object Object]". This flattens it to
    the field and what was wrong with it, and keeps the 422.
    """
    said = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        msg = err.get("msg", "invalid value")
        said.append(f"{where}: {msg}" if where else msg)
    return JSONResponse(status_code=422, content={"detail": "; ".join(said)})


app.include_router(api_router)


# Serve the built frontend when present (Docker). In local dev the directory may
# be absent and the Vite dev server is used instead, so guard on existence.
if STATIC_DIR.is_dir():
    assets = STATIC_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

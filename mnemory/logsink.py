"""Optional durable log file, so verdict lines outlive the container.

mnemory configures logging once, to stderr (``server.py``).  In Docker that
lands in the container's json-file log, which is destroyed when the container
is destroyed -- not rotated, not truncated, *gone*.  The write-side noise guard
logs every drop it takes, drops are irreversible (there is no un-retract
endpoint), and two audit corpora have already been lost this way: one to a
Watchtower update, one to a manual recreate during a guard change.

Setting ``MNEMORY_LOG_FILE`` to a path on a mounted volume moves that evidence
where the data already lives.  Stderr keeps everything it had; this only adds.

Kept in its own module, with no mnemory imports, so it is unit-testable without
qdrant or a service -- the whole point is a deterministic guarantee about a file.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

#: Environment variable naming the durable log file. Unset or blank = off,
#: which is the default: an agent writing to an arbitrary path should be a
#: deployment decision, not a surprise.
LOG_FILE_ENV = "MNEMORY_LOG_FILE"

#: 10 MB x 5 -- a few hundred MB at the absolute worst, and a guard verdict is
#: ~200 bytes, so that is millions of verdicts. Rotation is the cost of putting
#: logs on the same volume as the data, and this bounds it.
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

#: Same shape as server.py's basicConfig format, so grep and the existing log
#: readers work identically against either sink.
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def attach_file_handler(root: logging.Logger | None = None) -> RotatingFileHandler | None:
    """Attach a rotating file handler when ``MNEMORY_LOG_FILE`` names a usable path.

    Returns the handler attached (or already present), or ``None`` when logging
    to a file is disabled or the path cannot be used.  Never raises: losing the
    durable copy is bad, refusing to start the service over it is worse, so a
    bad path degrades to stderr-only with one warning.
    """
    raw = os.environ.get(LOG_FILE_ENV, "").strip()
    log = logging.getLogger("mnemory")
    if not raw:
        return None

    target = root if root is not None else logging.getLogger()
    path = Path(raw)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning(
            "%s=%s is unusable (%s: %s); continuing stderr-only",
            LOG_FILE_ENV,
            raw,
            type(exc).__name__,
            exc,
        )
        return None

    handler.setFormatter(logging.Formatter(FORMAT))

    # Idempotent: uvicorn reloads, test runs and a re-import of server.py all
    # reach this line again, and a duplicated handler duplicates every line in
    # the durable file -- which reads as evidence, so it must not lie.
    resolved = Path(handler.baseFilename).resolve()
    for existing in list(target.handlers):
        base = getattr(existing, "baseFilename", None)
        if base is not None and Path(base).resolve() == resolved:
            handler.close()
            log.debug("durable log handler already attached: %s", resolved)
            return existing

    target.addHandler(handler)
    log.info("durable log file attached: %s (max %d B x %d)", resolved, MAX_BYTES, BACKUP_COUNT)
    return handler

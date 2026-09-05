"""Load key/value pairs from a .env file into the process environment.

Written by hand rather than pulled in as a dependency: the format we need is a
dozen lines of parsing, and every extra install step is one more thing that can
fail on someone else's machine.

Existing environment variables always win. That way a value exported in a shell,
or injected by a CI runner or an IDE run configuration, is never silently
overridden by a stale file on disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def find_dotenv(start: Optional[Path] = None, filename: str = ".env") -> Optional[Path]:
    """Search for a .env file in this directory and its parents."""
    here = (start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def parse_dotenv(text: str) -> Dict[str, str]:
    """Parse the small subset of the format that matters.

    Supports comments, blank lines, an optional ``export`` prefix, and single or
    double quoted values. Anything more exotic belongs in a real shell script.
    """
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            # Strip a trailing inline comment only when unquoted.
            hash_pos = value.find(" #")
            if hash_pos != -1:
                value = value[:hash_pos].rstrip()
        out[key] = value
    return out


def load_dotenv(path: Optional[Path] = None, override: bool = False) -> Optional[Path]:
    """Load a .env file into ``os.environ``. Returns the file used, if any."""
    p = path or find_dotenv()
    if p is None or not p.is_file():
        return None
    try:
        values = parse_dotenv(p.read_text(encoding="utf-8-sig"))
    except OSError:
        return None
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return p

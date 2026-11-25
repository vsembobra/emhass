"""Lossless gzip storage behind legacy logical JSON/JSONL paths.

The compressed file is authoritative once atomically published. Legacy input is
still readable; training rows and quality flags are never aged out here.
"""

import gzip
import io
import json
import os
import tempfile
import threading
import zlib
from pathlib import Path

import weather_models

_LOCK = threading.RLock()


def compressed(path: Path) -> Path:
    return path.with_name(path.name + ".gz")


def exists(path: Path) -> bool:
    return compressed(path).is_file() or path.is_file()


def read_bytes(path: Path) -> bytes:
    # Retry gzip if migration publishes it between our first lookup and the
    # old-file open. Open file descriptors remain valid across replacement.
    for candidate, zipped in ((compressed(path), True), (path, False), (compressed(path), True)):
        try:
            raw = candidate.read_bytes()
        except FileNotFoundError:
            continue
        return gzip.decompress(raw) if zipped else raw
    raise FileNotFoundError(path)


def read_text(path: Path) -> str:
    return read_bytes(path).decode("utf-8")


def read_json(path: Path):
    return json.loads(read_text(path))


def read_frame(path: Path):
    import pandas as pd

    return pd.read_json(io.StringIO(read_text(path)), lines=True)


def _sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes(path: Path, raw: bytes) -> None:
    destination = compressed(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(raw, compresslevel=6, mtime=0)
    with _LOCK:
        remove_legacy = not (
            path.is_file() and compressed(path).is_file() and path.read_bytes() != read_bytes(path)
        )
        fd, name = tempfile.mkstemp(prefix=".emhass-shadow-" + destination.name + ".", dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
            _sync_directory(destination.parent)
            if remove_legacy:
                path.unlink(missing_ok=True)
                _sync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)


def write_text(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, value) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with _LOCK:
        text = read_text(path) if exists(path) else ""
        if text and not text.endswith("\n"):
            text += "\n"
        text += "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
        write_text(path, text)


def json_paths(directory: Path) -> list[Path]:
    """Unique logical paths, keeping .stem/.name compatible with dated reports."""
    paths = set(directory.glob("*.json"))
    paths.update(path.with_suffix("") for path in directory.glob("*.json.gz"))
    return sorted(paths)


def remove(path: Path) -> None:
    with _LOCK:
        compressed(path).unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def compact_existing(path: Path) -> bool:
    """Migrate a legacy file, or remove an exactly matching legacy duplicate.

    Different legacy contents are left untouched when gzip already exists.
    This protects unique data left by older writers or manual imports.
    """
    with _LOCK:
        if exists(path):
            canonical = read_bytes(path)
            for orphan in path.parent.glob(".emhass-shadow-" + compressed(path).name + ".*"):
                if not orphan.is_file() or orphan.is_symlink():
                    continue
                try:
                    duplicate = gzip.decompress(orphan.read_bytes()) == canonical
                except (OSError, EOFError, zlib.error):
                    duplicate = False
                if duplicate:
                    orphan.unlink()
        if not path.is_file() or path.is_symlink():
            return False
        raw = path.read_bytes()
        if compressed(path).exists():
            if read_bytes(path) != raw:
                return False
            path.unlink()
            _sync_directory(path.parent)
        else:
            _write_bytes(path, raw)
        return True


def compact_data(data: Path) -> int:
    """Known app-owned data only. Models, config, outage DB and user files stay."""
    roots = [data, *(data / "weather_models" / model for model in weather_models.MODEL_IDS if model != "auto")]
    count = 0
    for root in roots:
        if root.is_symlink():
            continue
        for name in ("hourly_audit.jsonl", "forecast_log.jsonl", "weather_previous_day1.jsonl", "metrics.jsonl"):
            count += compact_existing(root / name)
        for directory in (root / "shadow_reports", root / "weather_archive"):
            if directory.is_symlink():
                continue
            for path in json_paths(directory):
                count += compact_existing(path)
    return count

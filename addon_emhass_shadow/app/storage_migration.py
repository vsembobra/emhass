"""Consume retired app-owned share exports after verified, lossless preservation."""

import gzip
import hashlib
import json
import math
import re
import sqlite3
import zlib
from contextlib import closing
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import data_store as store
import weather_models

# Remove formatting whitespace only; string contents and numeric tokens stay exact.
_JSON_SPACE = re.compile(rb'"(?:\\.|[^"\\])*"|[ \t\r\n]+')


def _valid_json(raw: bytes, lines: bool) -> bool:
    try:
        for item in raw.splitlines() if lines else [raw]:
            if item.strip():
                json.loads(item)
        return bool(raw.strip()) or lines
    except (ValueError, UnicodeError):
        return False


def _equivalent(a: bytes, b: bytes, lines: bool) -> bool:
    if a == b:
        return True
    if not _valid_json(a, lines) or not _valid_json(b, lines):
        return False

    def normalized(raw):
        return _JSON_SPACE.sub(lambda match: match[0] if match[0].startswith(b'"') else b"", raw)

    return normalized(a) == normalized(b)


def _timestamp(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value)
        return stamp if stamp.tzinfo else None
    except ValueError:
        return None


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric(value: dict) -> bool:
    if not _timestamp(value.get("issue_utc")) or not isinstance(value.get("date"), str):
        return False
    for target in ("load", "pv"):
        score = value.get(target)
        if score is not None and (
            not isinstance(score, dict)
            or not all(_number(score.get(key)) for key in ("mae_kwh", "coverage_pct", "hours"))
        ):
            return False
    pvf = value.get("plan_vs_fact")
    if pvf:
        if not isinstance(pvf, dict) or not all(
            _number(pvf.get(key)) for key in ("planned_grid_kwh", "actual_grid_kwh")
        ):
            return False
        if "forecast_effect_kwh" in pvf and not all(
            _number(pvf.get(key)) for key in ("forecast_effect_kwh", "execution_effect_kwh")
        ):
            return False
    return True


def _can_restore(path: Path, raw: bytes) -> bool:
    if not _valid_json(raw, path.suffix == ".jsonl"):
        return False
    if path.suffix == ".jsonl":
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                return False
            if path.name == "metrics.jsonl" and not _metric(row):
                return False
            if path.name == "outages.jsonl":
                start, end = _timestamp(row.get("start_utc")), _timestamp(row.get("end_record_utc"))
                if (
                    not start
                    or not end
                    or end < start
                    or not isinstance(row.get("interrupted"), bool)
                ):
                    return False
        return True
    value = json.loads(raw)
    if not isinstance(value, dict):
        return False
    if "shadow_reports" in path.parts:
        advisory, reserve, load = (
            value.get(key) for key in ("advisory", "reserve", "load_forecast_kwh_24h")
        )
        return (
            bool(_timestamp(value.get("issue_utc")))
            and all(isinstance(item, dict) for item in (advisory, reserve, load))
            and all(
                _number(advisory.get(key))
                for key in ("recommended_floor_soc", "current_floor_soc", "delta_pp")
            )
            and isinstance(advisory.get("programs"), list)
            and all(_number(load.get(key)) for key in ("p10", "p50", "p90"))
        )
    if "weather_archive" in path.parts:
        hourly = value.get("hourly")
        if (
            not isinstance(hourly, dict)
            or not isinstance(hourly.get("time"), list)
            or not hourly["time"]
            or not all(_number(item) for item in hourly["time"])
        ):
            return False
        return all(
            isinstance(hourly.get(key), list)
            and len(hourly[key]) == len(hourly["time"])
            and all(item is None or _number(item) for item in hourly[key])
            for key in (
                "temperature_2m",
                "shortwave_radiation",
                "direct_normal_irradiance",
                "diffuse_radiation",
                "cloud_cover",
            )
        )
    if path.name == "retrain_state.json":
        models = value.get("models", {})
        return (
            (value.get("last_retrain_utc") is None or bool(_timestamp(value["last_retrain_utc"])))
            and isinstance(models, dict)
            and all(isinstance(info, dict) for info in models.values())
        )
    return True


def _archive(data: Path, relative: Path, raw: bytes, result: dict) -> None:
    archive = (
        data
        / "legacy_imports"
        / "share"
        / relative.parent
        / (hashlib.sha256(raw).hexdigest() + "_" + relative.name)
    )
    if not store.local_path(archive, data) or not store.local_path(store.compressed(archive), data):
        raise OSError("unsafe migration destination")
    if not store.exists(archive):
        store.write_bytes(archive, raw)
    if store.read_bytes(archive) != raw:
        raise OSError("archive verification failed")
    result["archived"] += 1


def _rows(raw: bytes) -> list[tuple[dict, bytes]]:
    rows = []
    for line in raw.splitlines():
        if line.strip():
            row = json.loads(line, parse_float=Decimal)
            if not isinstance(row, dict):
                raise ValueError("history row is not an object")
            rows.append((row, line))
    return rows


def _instant(row: dict) -> datetime:
    value = row.get("start_utc")
    if not isinstance(value, str):
        raise ValueError("missing UTC history timestamp")
    stamp = datetime.fromisoformat(value)
    return (stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)).astimezone(UTC)


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _covered(old: dict, current: dict) -> bool:
    return all(
        key == "start_utc"
        or (value is None and key not in current)
        or key in current
        and _same(value, current[key])
        for key, value in old.items()
    )


def _audit_seed(data: Path, relative: Path, raw: bytes, result: dict) -> None:
    target = data / "hourly_audit.jsonl"
    existing = store.read_bytes(target) if store.exists(target) else b""
    current = {_instant(row): row for row, _ in _rows(existing)}
    missing, conflicts = [], []
    for row, line in _rows(raw):
        stamp = _instant(row)
        if stamp not in current:
            missing.append(line)
            current[stamp] = row
        elif not _covered(row, current[stamp]):
            conflicts.append(line)
    if missing:
        combined = existing + (b"\n" if existing and not existing.endswith(b"\n") else b"")
        store.write_bytes(target, combined + b"\n".join(missing) + b"\n")
        result["imported_hours"] += len(missing)
    if conflicts:
        _archive(data, relative, b"\n".join(conflicts) + b"\n", result)
    conflicting_lines = set(conflicts)
    # Re-read committed data: every non-conflicting source observation must exist.
    saved = (
        {_instant(row): row for row, _ in _rows(store.read_bytes(target))}
        if store.exists(target)
        else {}
    )
    for row, line in _rows(raw):
        if line not in conflicting_lines and (
            _instant(row) not in saved or not _covered(row, saved[_instant(row)])
        ):
            raise OSError("history verification failed")


def _sqlite_covered(source: Path, target: Path) -> bool:
    if not target.is_file() or source.with_name(source.name + "-wal").exists():
        return False
    try:
        with (
            closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as old,
            closing(sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)) as current,
        ):
            tables = {
                row[0]
                for row in old.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if not tables or not tables <= {"outage_observations", "outage_events"}:
                return False
            for table in tables:
                if list(old.execute(f'PRAGMA table_info("{table}")')) != list(
                    current.execute(f'PRAGMA table_info("{table}")')
                ):
                    return False
                if not set(old.execute(f'SELECT * FROM "{table}"')) <= set(
                    current.execute(f'SELECT * FROM "{table}"')
                ):
                    return False
        return True
    except sqlite3.Error:
        return False


def _sources(share: Path) -> list[Path]:
    roots = [
        share,
        *(
            share / "weather_models" / model
            for model in weather_models.MODEL_IDS
            if model != "auto"
        ),
    ]
    files = []
    for root in roots:
        for name in ("metrics.jsonl", "retrain_state.json"):
            files.extend((store.compressed(root / name), root / name))
        directory = root / "shadow_reports"
        files.extend(sorted(directory.glob("*.json.gz")))
        files.extend(sorted(directory.glob("*.json")))
    seed = share / "seed"
    for name in ("hourly_audit.jsonl", "outages.jsonl", "outages.sqlite"):
        files.extend((store.compressed(seed / name), seed / name))
    files.extend(sorted((seed / "weather_archive").glob("*.json.gz")))
    files.extend(sorted((seed / "weather_archive").glob("*.json")))
    return [path for path in files if path.is_file() and store.local_path(path, share)]


def migrate_shared(data: Path, share: Path) -> dict:
    """No external consumers remain; unknown paths and unique versions are kept."""
    result = {"removed": 0, "restored": 0, "archived": 0, "imported_hours": 0, "kept": 0}
    if share.is_symlink() or (data.exists() and share.exists() and data.samefile(share)):
        return result
    for source in _sources(share):
        try:
            relative = source.relative_to(share)
            logical = relative.with_suffix("") if relative.suffix == ".gz" else relative
            seed = logical.parts[0] == "seed"
            target = data / (Path(*logical.parts[1:]) if seed else logical)
            if not store.local_path(target, data) or not store.local_path(
                store.compressed(target), data
            ):
                raise OSError("unsafe migration destination")
            encoded = source.read_bytes()
            raw = gzip.decompress(encoded) if relative.suffix == ".gz" else encoded
            if seed and logical.name == "hourly_audit.jsonl":
                _audit_seed(data, logical, raw, result)
            elif seed and logical.name == "outages.sqlite":
                # SQLite stays a live database. A WAL seed is not a closed snapshot.
                if source.with_name(source.name + "-wal").exists():
                    result["kept"] += 1
                    continue
                if not target.is_file():
                    result["kept"] += 1  # run.sh imports a fresh database first
                    continue
                if relative.suffix == ".gz" or not _sqlite_covered(source, target):
                    _archive(data, logical, raw, result)
            elif store.exists(target):
                if not _equivalent(raw, store.read_bytes(target), logical.suffix == ".jsonl"):
                    _archive(data, logical, raw, result)
            elif _can_restore(logical, raw):
                store.write_bytes(target, raw)
                if store.read_bytes(target) != raw:
                    raise OSError("import verification failed")
                result["restored"] += 1
            else:
                _archive(data, logical, raw, result)
            if source.read_bytes() != encoded:
                raise OSError("source changed during migration")
            source.unlink()
            store._sync_directory(source.parent)
            result["removed"] += 1
        except (OSError, EOFError, ValueError, zlib.error):
            result["kept"] += 1
    return result

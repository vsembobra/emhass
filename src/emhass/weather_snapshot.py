"""Capture a site's live Open-Meteo forecast in an append-only archive."""

import argparse
import json
import os
import re
from datetime import UTC, datetime
from hashlib import sha256
from http.client import HTTPException
from itertools import pairwise
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NoReturn
from urllib.parse import urlencode
from urllib.request import urlopen

FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
FORECAST_HOURS = 48
DEFAULT_MODEL = "best_match"
HOURLY_VARIABLES = (
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "snow_depth",
)


def _reject_json_constant(constant: str) -> NoReturn:
    raise ValueError(f"Invalid JSON constant in Open-Meteo response: {constant}")


def _valid_times_utc(forecast: object) -> list[str]:
    """Validate the provider's hourly arrays and produce their UTC index."""
    if not isinstance(forecast, dict) or forecast.get("error"):
        raise ValueError("Expected an Open-Meteo forecast object")
    hourly = forecast.get("hourly")
    units = forecast.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise ValueError("Open-Meteo response is missing hourly data or units")

    times = hourly.get("time")
    if (
        not isinstance(times, list)
        or not times
        or any(type(time) is not int or time % 3600 for time in times)
        or any(end - start != 3600 for start, end in pairwise(times))
    ):
        raise ValueError("Expected consecutive Unix timestamps at UTC hour boundaries")

    for variable in HOURLY_VARIABLES:
        values = hourly.get(variable)
        if not isinstance(values, list) or len(values) != len(times):
            raise ValueError(f"Missing or misaligned hourly values for {variable}")
        if any(value is not None and type(value) not in (int, float) for value in values):
            raise ValueError(f"Expected numeric or null hourly values for {variable}")
        if not isinstance(units.get(variable), str) or not units[variable]:
            raise ValueError(f"Missing hourly unit for {variable}")

    try:
        return [
            datetime.fromtimestamp(time, UTC).isoformat().replace("+00:00", "Z") for time in times
        ]
    except (OverflowError, OSError) as exc:
        raise ValueError("Open-Meteo timestamp is outside the supported date range") from exc


def capture_snapshot(
    *,
    site_id: str,
    latitude: float,
    longitude: float,
    archive_dir: Path,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Fetch one vintage and return its archive path, reusing an identical record."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", site_id):
        raise ValueError(
            "Site ID must start with a lowercase ASCII letter or digit and contain "
            "at most 64 lowercase ASCII letters, digits, hyphens, or underscores"
        )
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Latitude must be within ±90 and longitude within ±180 degrees")
    if not re.fullmatch(r"[a-z0-9_]+", model):
        raise ValueError("Provide a single Open-Meteo model ID")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_hours": FORECAST_HOURS,
        "models": model,
        "timezone": "UTC",
        "timeformat": "unixtime",
    }
    with urlopen(f"{FORECAST_ENDPOINT}?{urlencode(params)}", timeout=30) as response:
        body = response.read()
        received_at = datetime.now(UTC)

    forecast = json.loads(body.decode("utf-8"), parse_constant=_reject_json_constant)
    valid_times = _valid_times_utc(forecast)
    request = {"endpoint": FORECAST_ENDPOINT, "params": params}
    # API processing duration changes on identical fetches. Keep it in the raw response only.
    forecast_identity = {
        key: value for key, value in forecast.items() if key != "generationtime_ms"
    }
    identity = json.dumps([request, forecast_identity], sort_keys=True, allow_nan=False)
    digest = sha256(identity.encode("utf-8")).hexdigest()
    metadata = {
        "schema_version": 1,
        "site_id": site_id,
        "provider": "Open-Meteo",
        "request": request,
        "received_at_utc": received_at.isoformat().replace("+00:00", "Z"),
        "valid_times_utc": valid_times,
    }
    # Embed the validated JSON body directly to preserve even its original number formatting.
    header = json.dumps(metadata, indent=2).removesuffix("\n}").encode("utf-8")
    content = header + b',\n  "response": ' + body + b"\n}\n"

    directory = archive_dir / site_id
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}.json"
    with NamedTemporaryFile(dir=directory, prefix=".snapshot-", delete_on_close=False) as staged:
        staged.write(content)
        staged.flush()
        os.fsync(staged.fileno())
        staged.close()
        # Link a complete file without replacing an existing vintage, even across processes.
        try:
            os.link(staged.name, destination)
        except FileExistsError:
            pass
    return destination


def main(argv: list[str] | None = None) -> int:
    """Run the one-shot snapshot command."""
    parser = argparse.ArgumentParser(prog="python -m emhass.weather_snapshot", description=__doc__)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--latitude", required=True, type=float)
    parser.add_argument("--longitude", required=True, type=float)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Open-Meteo model ID (best_match)")
    args = parser.parse_args(argv)
    try:
        destination = capture_snapshot(**vars(args))
    except (OSError, HTTPException, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

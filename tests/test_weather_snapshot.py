"""Tests for the standalone Open-Meteo archive using mocked HTTP responses."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from emhass import weather_snapshot


@pytest.fixture
def forecast():
    """Provider JSON with hours spanning the October clock change in Europe."""
    units = {
        "temperature_2m": "°C",
        "shortwave_radiation": "W/m²",
        "direct_radiation": "W/m²",
        "diffuse_radiation": "W/m²",
        "direct_normal_irradiance": "W/m²",
        "cloud_cover": "%",
        "cloud_cover_low": "%",
        "cloud_cover_mid": "%",
        "cloud_cover_high": "%",
        "snow_depth": "m",
    }
    start = int(datetime(2026, 10, 25, tzinfo=UTC).timestamp())
    return {
        "latitude": 52.52,
        "longitude": 13.42,
        "elevation": 44,
        "generationtime_ms": 2.125,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "timezone_abbreviation": "UTC",
        "hourly_units": {"time": "unixtime", **units},
        "hourly": {
            "time": [start + i * 3600 for i in range(48)],
            **{variable: [0.0] * 48 for variable in units},
            "temperature_2m": [-2.5] * 48,
            "snow_depth": [None] + [0.1] * 47,
        },
    }


@pytest.fixture
def site(tmp_path):
    return {"site_id": "home-1", "latitude": 52.51, "longitude": 13.41, "archive_dir": tmp_path}


@pytest.fixture
def cli_args(site):
    return [
        "--site-id",
        site["site_id"],
        "--latitude",
        str(site["latitude"]),
        "--longitude",
        str(site["longitude"]),
        "--archive-dir",
        str(site["archive_dir"]),
    ]


def test_cli_archives_raw_response_and_receipt_and_valid_times(site, cli_args, forecast, capsys):
    body = json.dumps(forecast, ensure_ascii=False, indent=1).encode("utf-8")
    before = datetime.now(UTC)
    with patch.object(weather_snapshot, "urlopen", return_value=BytesIO(body)) as fetch:
        assert weather_snapshot.main(cli_args) == 0
    after = datetime.now(UTC)

    request = urlsplit(fetch.call_args.args[0])
    assert (request.scheme, request.netloc, request.path) == (
        "https",
        "api.open-meteo.com",
        "/v1/forecast",
    )
    params = parse_qs(request.query)
    assert params == {
        "latitude": ["52.51"],
        "longitude": ["13.41"],
        "hourly": [",".join(k for k in forecast["hourly_units"] if k != "time")],
        "forecast_hours": ["48"],
        "models": ["best_match"],
        "timezone": ["UTC"],
        "timeformat": ["unixtime"],
    }
    path = Path(capsys.readouterr().out.strip())
    assert path.parent == site["archive_dir"] / "home-1"
    content = path.read_bytes()
    assert body in content
    record = json.loads(content)
    assert record["schema_version"] == 1
    assert record["site_id"] == "home-1"
    assert record["provider"] == "Open-Meteo"
    assert record["request"]["endpoint"] == "https://api.open-meteo.com/v1/forecast"
    assert record["request"]["params"]["models"] == "best_match"
    assert record["response"] == forecast
    assert before <= datetime.fromisoformat(record["received_at_utc"]) <= after
    assert "issued_at_utc" not in record
    assert len(record["valid_times_utc"]) == 48
    assert record["valid_times_utc"][:3] == [
        "2026-10-25T00:00:00Z",
        "2026-10-25T01:00:00Z",
        "2026-10-25T02:00:00Z",
    ]
    assert record["valid_times_utc"][-1] == "2026-10-26T23:00:00Z"


def test_retry_retains_the_first_snapshot_when_only_processing_duration_changes(site, forecast):
    first_body = json.dumps(forecast).encode()
    forecast["generationtime_ms"] = 8.125
    retry_body = json.dumps(forecast, sort_keys=True).encode()
    with patch.object(
        weather_snapshot, "urlopen", side_effect=[BytesIO(first_body), BytesIO(retry_body)]
    ):
        first = weather_snapshot.capture_snapshot(**site)
        first_content = first.read_bytes()
        second = weather_snapshot.capture_snapshot(**site)

    assert second == first
    assert list(first.parent.iterdir()) == [first]
    assert first.read_bytes() == first_content
    assert first_body in first_content


def test_changed_forecast_and_requested_model_get_new_vintages(site, forecast):
    first_body = json.dumps(forecast).encode()
    forecast["hourly"]["cloud_cover"][0] = 10
    revised_body = json.dumps(forecast).encode()
    with patch.object(
        weather_snapshot,
        "urlopen",
        side_effect=[
            BytesIO(first_body),
            BytesIO(revised_body),
            BytesIO(revised_body),
        ],
    ):
        first = weather_snapshot.capture_snapshot(**site)
        revised = weather_snapshot.capture_snapshot(**site)
        other_model = weather_snapshot.capture_snapshot(**site, model="icon_global")

    assert len({first, revised, other_model}) == 3
    assert first_body in first.read_bytes()
    assert revised_body in revised.read_bytes()
    assert json.loads(other_model.read_bytes())["request"]["params"]["models"] == "icon_global"


def test_concurrent_writes_publish_one_complete_file(site, forecast, monkeypatch):
    body = json.dumps(forecast).encode()
    barrier = Barrier(2)
    real_link = weather_snapshot.os.link

    def concurrent_link(*args):
        barrier.wait(timeout=5)
        return real_link(*args)

    monkeypatch.setattr(weather_snapshot.os, "link", concurrent_link)
    with (
        patch.object(weather_snapshot, "urlopen", side_effect=lambda *_a, **_k: BytesIO(body)),
        ThreadPoolExecutor(2) as pool,
    ):
        calls = [pool.submit(weather_snapshot.capture_snapshot, **site) for _ in range(2)]
        first, second = (call.result() for call in calls)

    assert first == second
    assert list(first.parent.iterdir()) == [first]
    assert json.loads(first.read_bytes())["response"] == forecast


@pytest.mark.parametrize(
    "override",
    [
        {"site_id": "../home"},
        {"site_id": "home/a"},
        {"site_id": "home\\a"},
        {"site_id": ""},
        {"latitude": 91},
        {"longitude": float("nan")},
        {"model": "a,b"},
    ],
)
def test_invalid_site_or_request_args_fail_before_fetch(site, override):
    with (
        patch.object(weather_snapshot, "urlopen", side_effect=AssertionError("unexpected fetch")),
        pytest.raises(ValueError),
    ):
        weather_snapshot.capture_snapshot(**(site | override))

    assert not list(site["archive_dir"].iterdir())


@pytest.mark.parametrize(
    "invalid", ["json", "missing series", "missing unit", "short series", "time", "NaN", "bool"]
)
def test_invalid_response_does_not_create_a_snapshot(site, forecast, invalid):
    if invalid == "missing series":
        del forecast["hourly"]["snow_depth"]
    elif invalid == "missing unit":
        del forecast["hourly_units"]["snow_depth"]
    elif invalid == "short series":
        forecast["hourly"]["snow_depth"].pop()
    elif invalid == "time":
        forecast["hourly"]["time"][1] = forecast["hourly"]["time"][0]
    elif invalid == "NaN":
        forecast["hourly"]["snow_depth"][0] = float("nan")
    elif invalid == "bool":
        forecast["hourly"]["snow_depth"][0] = False
    body = b"{" if invalid == "json" else json.dumps(forecast).encode()

    with (
        patch.object(weather_snapshot, "urlopen", return_value=BytesIO(body)),
        pytest.raises(ValueError),
    ):
        weather_snapshot.capture_snapshot(**site)

    assert not list(site["archive_dir"].iterdir())


def test_cli_exits_on_http_failure(site, cli_args, capsys):
    with (
        patch.object(weather_snapshot, "urlopen", side_effect=URLError("network unavailable")),
        pytest.raises(SystemExit) as raised,
    ):
        weather_snapshot.main(cli_args)

    assert raised.value.code == 1
    output = capsys.readouterr()
    assert "network unavailable" in output.err
    assert output.out == ""
    assert not list(site["archive_dir"].iterdir())


def test_failed_publish_cleans_up_the_temporary_file(site, forecast):
    body = json.dumps(forecast).encode()
    with (
        patch.object(weather_snapshot, "urlopen", return_value=BytesIO(body)),
        patch.object(weather_snapshot.os, "link", side_effect=OSError("publication failed")),
        pytest.raises(OSError, match="publication failed"),
    ):
        weather_snapshot.capture_snapshot(**site)

    assert not list((site["archive_dir"] / site["site_id"]).iterdir())

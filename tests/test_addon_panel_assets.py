"""The panel's chart library is bundled and served by the add-on itself."""

import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import main  # noqa: E402


@pytest.mark.parametrize("asset", ["panel_state.js", "panel_ui.js"])
def test_panel_modules_are_served_locally(asset):
    panel = (ADDON / "app" / "panel.html").read_text()
    assert f'<script src="{asset}"></script>' in panel
    server = ThreadingHTTPServer(("127.0.0.1", 0), main.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/{asset}"
        with urllib.request.urlopen(url, timeout=10) as response:
            assert response.headers["Content-Type"].startswith("application/javascript")
            assert response.read() == (ADDON / "app" / asset).read_bytes()
    finally:
        server.shutdown()
        server.server_close()


def test_echarts_is_bundled_and_served_offline():
    panel = (ADDON / "app" / "panel.html").read_text(encoding="utf-8")
    assert '<script src="vendor/echarts.min.js"></script>' in panel
    assert "cdn" not in panel.lower()  # no internet needed to draw the charts
    server = ThreadingHTTPServer(("127.0.0.1", 0), main.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/vendor/echarts.min.js"
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read()
            assert response.headers["Content-Type"].startswith("application/javascript")
            assert "max-age" in response.headers["Cache-Control"]
        assert b"Apache License" in body[:2000] and b'version="5.5.1"' in body
    finally:
        server.shutdown()

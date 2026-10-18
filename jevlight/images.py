"""Synchronized camera frames from JevLight's existing Three.js scene."""

from __future__ import annotations

import base64
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Sequence

from jevlight.viz_server import (
    NetworkModel, PassiveRuntime, start_viz_server, vehicle_class_index,
)


def camera_snapshot(conn: Any) -> Dict[str, Any]:
    """Read a frozen SUMO snapshot without replacing the env's subscriptions.

    sumolib lane shapes and TraCI positions both use native network coordinates.
    The renderer consumes this geometry only; it never sends it to the model.
    """
    vehicles = []
    for vid in conn.vehicle.getIDList():
        x, y, z = conn.vehicle.getPosition3D(vid)
        r, g, b, _alpha = conn.vehicle.getColor(vid)
        vehicles.append([
            vid, x, y, z, conn.vehicle.getAngle(vid), conn.vehicle.getSpeed(vid),
            r, g, b, vehicle_class_index(conn.vehicle.getVehicleClass(vid)),
        ])
    return {
        "t": float(conn.simulation.getTime()),
        "vehicles": vehicles,
        "tls": {
            tid: conn.trafficlight.getRedYellowGreenState(tid)
            for tid in conn.trafficlight.getIDList()
        },
    }


class ImageRenderer:
    """One headless Chromium instance for an experiment's camera composites."""

    def __init__(self, roadnet_path: str, output_dir: Path):
        self.network = NetworkModel.load(roadnet_path)
        self.output_dir = Path(output_dir)
        self._stack = ExitStack()

    def __enter__(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Image decisions need pip install -r requirements-images.txt "
                "and python -m playwright install chromium"
            ) from exc
        try:
            self.runtime = PassiveRuntime()
            self.server, url = start_viz_server(
                self.runtime, self.network.payload, host="127.0.0.1", port=0,
            )
            self._stack.callback(self.server.server_close)
            self._stack.callback(self.server.shutdown)
            self._stack.callback(self.runtime.close)
            playwright = self._stack.enter_context(sync_playwright())
            launch_args = {
                "headless": True,
                "args": ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
            }
            try:
                browser = playwright.chromium.launch(**launch_args)
            except Exception as bundled_error:
                # A machine may have Chrome installed but not Playwright's
                # 200 MiB browser bundle. Use it as a local fallback.
                try:
                    browser = playwright.chromium.launch(channel="chrome", **launch_args)
                except Exception:
                    raise bundled_error
            self._stack.callback(browser.close)
            self.page = browser.new_page(viewport={"width": 1024, "height": 768})
            self.page.goto(url + "/?capture=1", wait_until="load")
            self.page.wait_for_function("typeof window.captureJevFrame === 'function'", timeout=60000)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self
        except Exception:
            self._stack.close()
            raise

    def capture(self, conn: Any, intersection_ids: Sequence[str], step: int) -> Dict[str, Path]:
        frame = camera_snapshot(conn)
        self.page.evaluate("frame => window.setJevFrame(frame)", frame)
        paths = {}
        for index, intersection_id in enumerate(intersection_ids):
            data_url = self.page.evaluate(
                "id => window.captureJevFrame(id)", intersection_id,
            )
            # Numeric filenames also work for networks whose IDs contain path separators.
            path = self.output_dir / f"step_{step:06d}_junction_{index:04d}.jpg"
            path.write_bytes(base64.b64decode(data_url.split(",", 1)[1], validate=True))
            paths[intersection_id] = path
        return paths

    def __exit__(self, *exc):
        self._stack.close()

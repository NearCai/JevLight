"""VLALight's TransSimHub/Panda3D cameras, connected to Jev's image input."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw

SCENES = {("Jinan", "3_4"): "Jinan", ("Hangzhou", "4_4"): "Hangzhou",
          ("NewYork", "28_7"): "NewYork", ("NewYork", "16x3"): "NewYork16x3"}


def tls_camera_info(conn, tls_ids):
    """Use the same stop lines and heading ordering as VLALight's SUMOEnv."""
    info, labels, centers = {}, {}, {}
    for tls_id in tls_ids:
        lanes = [links[0][0] for links in conn.trafficlight.getControlledLinks(tls_id) if links]
        roads = sorted({conn.lane.getEdgeID(lane) for lane in lanes})
        headings = {road: conn.edge.getAngle(road) for road in roads}
        stops = {road: [conn.lane.getShape(lane)[-1] for lane in lanes
                        if conn.lane.getEdgeID(lane) == road] for road in roads}
        info[tls_id] = {"in_roads_heading": headings, "in_road_stop_line": stops}
        road_labels = {}
        for lane in lanes:
            points = conn.lane.getShape(lane)
            dx, dy = points[-1][0] - points[0][0], points[-1][1] - points[0][1]
            # SUMO phase labels describe vehicle travel direction, not camera bearing.
            index = round(math.atan2(dy, dx) / (math.pi / 2)) % 4
            road_labels[conn.lane.getEdgeID(lane)] = ("EASTBOUND", "NORTHBOUND", "WESTBOUND", "SOUTHBOUND")[index]
        labels[tls_id] = [road_labels[road] for road in sorted(roads, key=headings.get)]
        if len(labels[tls_id]) != 4 or len(set(labels[tls_id])) != 4:
            raise ValueError(f"Expected four distinct compass approaches: {tls_id}")
        points = [point for road_stops in stops.values() for point in road_stops]
        centers[tls_id] = tuple(sum(p[axis] for p in points) / len(points) for axis in (0, 1))
    return {"tls": info}, labels, centers


def compose_views(views, labels):
    """Preserve VLALight's 40%/30% horizontal crop and the image aspect ratio."""
    canvas = Image.new("RGB", (768, 1440))
    for index, (view, label) in enumerate(zip(views, labels)):
        image = Image.fromarray(view).convert("RGB")
        width, height = image.size
        image = image.crop((int(width * .4), 0, int(width * .7), height))
        image = image.resize((384, 720), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 384, 28), fill="black")
        draw.text((8, 4), label, fill="white", font_size=19)
        canvas.paste(image, ((index % 2) * 384, (index // 2) * 720))
    return canvas


class VLALightRenderer:
    def __init__(self, source_root, dataset_config, roadnet, output_dir, batch_size=4):
        if batch_size < 1:
            raise ValueError("render_batch_size must be positive")
        self.source_root = Path(source_root).resolve()
        self.roadnet = Path(roadnet).resolve()
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        key = (dataset_config["template"], dataset_config["road_net"])
        if key not in SCENES:
            raise ValueError(f"No VLALight GLB assets registered for {key}")
        self.scene_dir = self.source_root / "TransSimHub/examples" / SCENES[key] / "3d_assets"
        self.renderer = None
        self.metadata = {"renderer": "VLALight/TransSimHub/Panda3D", "scene": SCENES[key],
                         "preset": "1080P", "camera_height_m": 30, "radius_m": 200,
                         "tile_size": [384, 720], "composite_size": [768, 1440],
                         "crop_left": .4, "crop_right": .3, "batch_size": batch_size}

    def __enter__(self):
        if not self.scene_dir.is_dir():
            raise FileNotFoundError("VLALight assets missing; run python scripts/fetch_vlalight_renderer.py")
        manifest_path = self.source_root.parent / "renderer-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Pinned renderer manifest missing; use scripts/fetch_vlalight_renderer.py")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.metadata.update(repository=manifest["repository"], revision=manifest["revision"])
        sys.path.insert(0, str(self.source_root))
        from TransSimHub.tshub.tshub_env3d.vis3d_renderer.tshub_render import TSHubRenderer
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        self.renderer_class = TSHubRenderer
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def capture(self, conn, tls_ids, step):
        tls_ids = list(tls_ids)
        if self.renderer is None:
            init, self.labels, self.centers = tls_camera_info(conn, tls_ids)
            self.renderer = self.renderer_class(
                simid="sumo", sensor_config={"tls": {tls: {
                    "sensor_types": ["junction_front_all"], "tls_camera_height": 30
                } for tls in tls_ids}}, preset="1080P", resolution=1.,
                scenario_glb_dir=str(self.scene_dir), vehicle_model="low",
                render_mode="offscreen", rendering_backend="pandagl", show_buildings=False,
                tls_batch_size=self.batch_size, keep_batch_sensors=False,
                reuse_batch_sensors=True, step_task_manager=False,
                netxml_path=str(self.roadnet), show_arrows=False,
            )
            self.renderer.reset(init)
        # Read once without modifying subscriptions or advancing the simulation.
        snapshot = {vid: {"vehicle_type": conn.vehicle.getTypeID(vid),
                          "position": conn.vehicle.getPosition(vid),
                          "heading": conn.vehicle.getAngle(vid),
                          "length": conn.vehicle.getLength(vid)}
                    for vid in conn.vehicle.getIDList()}
        paths = {}
        batches = self.renderer.get_batch_info()["total_batches"]
        for batch in range(batches):
            self.renderer.switch_to_batch(batch)
            current_ids = self.renderer.get_current_batch_tls_ids()
            vehicles = {vid: value for vid, value in snapshot.items() if any(
                (value["position"][0] - self.centers[tls][0]) ** 2 +
                (value["position"][1] - self.centers[tls][1]) ** 2 <= 200 ** 2
                for tls in current_ids)}
            self.renderer.step({"vehicle": vehicles}, should_count_vehicles=False)
            # The original Windows backend draws asynchronously; read after the
            # fence so camera reuse cannot return the previous junction's pixels.
            self.renderer._showbase_instance.graphicsEngine.syncFrame()
            sync = self.renderer.scene_sync
            images = sync.collect_sensors(sync._current_batch_tls_elements())
            for tls in current_ids:
                views = [images[f"{tls}_{index}"]["junction_front_all"] for index in range(4)]
                image = compose_views(views, self.labels[tls])
                path = self.output_dir / f"step_{step:06d}_{tls}.jpg"
                image.save(path, quality=90)
                paths[tls] = path
        if set(paths) != set(tls_ids):
            raise RuntimeError("Incomplete VLALight image batch")
        return paths

    def __exit__(self, *_exc):
        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None

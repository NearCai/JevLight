"""3D visualizer server core: SUMO frames in, SSE/HTTP out.

Serves the ``viz/`` frontend (three.js) and streams live simulation state
so a browser can watch a SUMO network in 3D — either a standalone
simulation (``scripts/serve_3d.py``) or the JevLight experiment itself
(``run_jevlight.py --visualize``).

Two runtimes share one HTTP layer:

- :class:`SumoRuntime` — owns a traci connection and steps SUMO on its own
  thread (standalone mode).  Play/pause/step/speed command the thread.
- :class:`PassiveRuntime` — no thread, no own connection: the host process
  (the JevLight control loop) calls :meth:`PassiveRuntime.publish` with its
  traci connection after each ``env.step``, and consults
  :meth:`PassiveRuntime.wait_until_running` between steps, so the page's
  play/pause/step buttons gate the experiment itself.  ``speed`` is
  display-only here: the host paces the simulation, not this runtime.

Only the standard library plus ``requests``-free pure Python is used.  traci
positions are network-offset coordinates; the sumolib offset is subtracted
so vehicle frames and lane shapes share one coordinate frame.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIZ_HOST = "127.0.0.1"
DEFAULT_VIZ_PORT = 8300
DEFAULT_VIZ_DIR = REPO_ROOT / "viz"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}

# traci variable ids (traci.constants), with literal fallbacks so this
# module imports even without a traci installation.
try:  # pragma: no cover - exercised whenever traci is installed
    import traci.constants as tc

    VAR_POSITION3D = tc.VAR_POSITION3D
    VAR_POSITION = tc.VAR_POSITION
    VAR_ANGLE = tc.VAR_ANGLE
    VAR_SPEED = tc.VAR_SPEED
    VAR_COLOR = tc.VAR_COLOR
    VAR_VEHICLECLASS = tc.VAR_VEHICLECLASS
except Exception:  # pragma: no cover
    VAR_POSITION3D = 0x39
    VAR_POSITION = 0x42
    VAR_ANGLE = 0x43
    VAR_SPEED = 0x40
    VAR_COLOR = 0x45
    VAR_VEHICLECLASS = 0x73

SUBSCRIPTION_VARS = (
    VAR_POSITION3D,
    VAR_ANGLE,
    VAR_SPEED,
    VAR_COLOR,
    VAR_VEHICLECLASS,
)

# SUMO vClass -> size class used by the 3D vehicle models:
# 0 = passenger car, 1 = long vehicle (bus/truck/trailer/delivery),
# 2 = two-wheeler (bicycle/moped/motorcycle).
VEHICLE_CLASS_IDS = {
    "bus": 1,
    "truck": 1,
    "trailer": 1,
    "delivery": 1,
    "bicycle": 2,
    "moped": 2,
    "motorcycle": 2,
}


def vehicle_class_index(vclass: Any) -> int:
    """Map a SUMO vehicle class string to the 3D model size class."""
    return VEHICLE_CLASS_IDS.get(str(vclass or "").strip().lower(), 0)


# ---------------------------------------------------------------------- #
# Road-network model
# ---------------------------------------------------------------------- #


class NetworkModel:
    """Lane geometry and traffic-light positions extracted with sumolib."""

    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload

    @classmethod
    def load(cls, net_path: str) -> "NetworkModel":
        import sumolib

        net = sumolib.net.readNet(net_path)
        offset_x, offset_y = net.getLocationOffset()
        lanes: List[Dict[str, Any]] = []
        for edge in net.getEdges(withInternal=False):
            for lane in edge.getLanes():
                shape = [
                    [round(float(x), 2), round(float(y), 2)]
                    for x, y in lane.getShape()
                ]
                if len(shape) < 2:
                    continue
                lanes.append({
                    "id": lane.getID(),
                    "width": round(float(lane.getWidth()), 2),
                    "shape": shape,
                })
        traffic_lights: List[Dict[str, Any]] = []
        for tls in net.getTrafficLights():
            tls_id = tls.getID()
            try:
                x, y = net.getNode(tls_id).getCoord()
            except Exception:
                continue
            # Incoming-lane -> signal-link indices, so the frontend can
            # color each approach head by the exact chars of its own lane
            # in the live SUMO state string.
            lane_links: Dict[str, List[int]] = {}
            for connection in tls.getConnections():
                lane_in, _lane_out, link_index = connection[0], connection[1], connection[2]
                lane_links.setdefault(str(lane_in.getID()), []).append(int(link_index))
            traffic_lights.append({
                "id": tls_id,
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "lane_links": lane_links,
            })
        (min_x, min_y), (max_x, max_y) = net.getBBoxXY()
        junctions: List[Dict[str, Any]] = []
        for node in net.getNodes():
            try:
                shape = node.getShape() or []
            except Exception:
                continue
            if len(shape) < 3:
                continue  # dead-ends have no area to fill
            poly = [[round(float(x), 2), round(float(y), 2)] for x, y in shape]
            xs = [point[0] for point in poly]
            ys = [point[1] for point in poly]
            if max(xs) - min(xs) < 4 and max(ys) - min(ys) < 4:
                continue  # degenerate sliver
            junctions.append({"shape": poly})
        return cls({
            "offset": [float(offset_x), float(offset_y)],
            "bounds": {
                "min": [round(float(min_x), 1), round(float(min_y), 1)],
                "max": [round(float(max_x), 1), round(float(max_y), 1)],
            },
            "lanes": lanes,
            "tls": traffic_lights,
            "junctions": junctions,
        })


# ---------------------------------------------------------------------- #
# Frame capture (shared by both runtimes)
# ---------------------------------------------------------------------- #


def subscribe_vehicles(conn: Any, subscribed: set) -> None:
    """Subscribe every present vehicle once; results arrive per step."""
    for vid in conn.vehicle.getIDList():
        if vid not in subscribed:
            conn.vehicle.subscribe(vid, list(SUBSCRIPTION_VARS))
            subscribed.add(vid)


def capture_frame(conn: Any, offset: Tuple[float, float]) -> Dict[str, Any]:
    """One snapshot: vehicle rows + traffic-light states, net coordinates.

    Vehicle row: ``[id, x, y, z, angle, speed, r, g, b, class_idx]`` —
    ``class_idx`` picks the 3D model size (car / long vehicle / two-wheeler).
    """
    results = conn.vehicle.getAllSubscriptionResults() or {}
    vehicles = []
    for vid, data in results.items():
        pos = data.get(VAR_POSITION3D) or data.get(VAR_POSITION) or (0, 0, 0)
        angle = float(data.get(VAR_ANGLE, 0.0) or 0.0)
        speed = float(data.get(VAR_SPEED, 0.0) or 0.0)
        color = data.get(VAR_COLOR) or (255, 255, 0, 255)
        vehicles.append([
            vid,
            round(float(pos[0]) - offset[0], 2),
            round(float(pos[1]) - offset[1], 2),
            round(float(pos[2] if len(pos) > 2 else 0.0), 2),
            round(angle, 1),
            round(speed, 2),
            int(color[0]),
            int(color[1]),
            int(color[2]),
            vehicle_class_index(data.get(VAR_VEHICLECLASS)),
        ])
    tls_states = {}
    for tid in conn.trafficlight.getIDList():
        tls_states[tid] = conn.trafficlight.getRedYellowGreenState(tid)
    return {
        "t": float(conn.simulation.getTime()),
        "vehicles": vehicles,
        "tls": tls_states,
    }


class _RuntimeBase:
    """State shared by both runtimes: frame store, commands, shutdown."""

    def __init__(self, step_length: float, offset: Tuple[float, float]):
        self.step_length = max(0.01, float(step_length))
        self.speed = 1.0
        self.offset = offset
        self.playing = True
        self.stop_event = threading.Event()
        self.lock = threading.Condition()
        self.frame: Dict[str, Any] = {"seq": 0, "t": 0.0, "vehicles": [], "tls": {}}
        self._seq = 0
        self._subscribed: set = set()

    def command(self, **payload: Any) -> None:
        action = payload.get("action")
        with self.lock:
            self._apply_command(action, payload)
            self.lock.notify_all()

    def _apply_command(self, action: Optional[str], payload: Dict[str, Any]) -> None:
        if action == "play":
            self.playing = True
            self._set_paused(False)
        elif action == "pause":
            self.playing = False
            self._set_paused(True)
        elif action == "speed":
            self.speed = max(0.01, float(payload.get("speed", 1.0)))
        elif action == "quit":
            self.stop_event.set()

    def _set_paused(self, paused: bool) -> None:  # overridden where relevant
        pass

    def close(self) -> None:
        self.stop_event.set()
        with self.lock:
            self.lock.notify_all()

    def _store(self, frame: Dict[str, Any]) -> None:
        """Store a frame tagged with the current step count.

        ``_seq`` counts simulation steps (not publishes), so a paused
        runtime republishing identical state never bumps the sequence and
        SSE writers stay quiet.
        """
        frame["seq"] = self._seq
        with self.lock:
            self.frame = frame
            self.lock.notify_all()


# ---------------------------------------------------------------------- #
# Standalone runtime: own connection, own stepping thread
# ---------------------------------------------------------------------- #


class SumoRuntime(_RuntimeBase):
    """Steps SUMO on one thread and publishes snapshots for SSE writers.

    ``conn`` is a traci connection (or a duck-typed stand-in for tests).
    The network ``offset`` is subtracted from vehicle positions so frames
    share the NetworkModel coordinate frame.
    """

    def __init__(
        self,
        conn: Any,
        step_length: float = 1.0,
        speed: float = 1.0,
        offset: Tuple[float, float] = (0.0, 0.0),
        max_steps: int = 0,
        playing: bool = True,
        restart=None,
    ):
        super().__init__(step_length, offset)
        self.conn = conn
        self.speed = max(0.01, float(speed))
        self.max_steps = max(0, int(max_steps))
        self.playing = bool(playing)
        self.restart = restart
        self.ended = False
        self.commands: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._step_requests = 0

    # -- control (any thread) ------------------------------------------- #

    def command(self, **payload: Any) -> None:  # noqa: D102 - adds step
        if payload.get("action") == "step":
            self._step_requests += 1
            return
        super().command(**payload)

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                stepped = False
                try:
                    started = time.time()
                    self._drain_commands()
                    if self.playing or self._step_requests > 0:
                        if not self.playing:
                            self._step_requests -= 1
                        self._step()
                        stepped = True
                        if self.max_steps and self._seq >= self.max_steps:
                            self.playing = False
                    self._publish()
                except Exception as exc:
                    # SUMO died (simulation end / crash).  With a restart
                    # hook (standalone loop mode) relaunch it; otherwise
                    # flag the final frame and stop.
                    if self.restart and not self.stop_event.is_set():
                        print(
                            f"[viz] SUMO ended ({type(exc).__name__}: {exc}); "
                            "restarting",
                            flush=True,
                        )
                        try:
                            self.conn = self.restart()
                            self._subscribed.clear()
                            continue
                        except Exception as restart_exc:
                            print(f"[viz] restart failed: {restart_exc}", flush=True)
                    self.ended = True
                    with self.lock:
                        self.frame = {**self.frame, "ended": True}
                        self.lock.notify_all()
                    return
                if stepped:
                    target = self.step_length / self.speed
                    elapsed = time.time() - started
                    if elapsed < target:
                        time.sleep(target - elapsed)
                else:
                    time.sleep(0.05)
        finally:
            close = getattr(self.conn, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

    def _drain_commands(self) -> None:
        while True:
            try:
                payload = self.commands.get_nowait()
            except queue.Empty:
                return
            action = payload.get("action")
            if action == "step":
                self._step_requests += 1
            else:
                with self.lock:
                    self._apply_command(action, payload)
                    self.lock.notify_all()

    def command_async(self, **payload: Any) -> None:
        """Queue a command for the simulation thread (used by HTTP)."""
        self.commands.put(payload)

    def _step(self) -> None:
        subscribe_vehicles(self.conn, self._subscribed)
        self.conn.simulationStep()
        self._seq += 1

    def _publish(self) -> None:
        self._store(capture_frame(self.conn, self.offset))


# ---------------------------------------------------------------------- #
# Passive runtime: the host process steps the simulation
# ---------------------------------------------------------------------- #


class PassiveRuntime(_RuntimeBase):
    """Publishes frames for a simulation the host process steps.

    The host calls :meth:`publish` with its traci connection after each
    simulation step, and :meth:`wait_until_running` between steps — so the
    page's pause/step buttons gate the host's control loop.  ``speed`` is
    display-only: the host owns the pace.
    """

    def __init__(
        self,
        step_length: float = 1.0,
        offset: Tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(step_length, offset)
        self.paused = False
        self._step_requests = 0

    def _set_paused(self, paused: bool) -> None:
        self.paused = paused

    def _apply_command(self, action: Optional[str], payload: Dict[str, Any]) -> None:
        super()._apply_command(action, payload)
        if action == "step":
            self._step_requests += 1

    def command(self, **payload: Any) -> None:  # noqa: D102 - notifies waiters
        with self.lock:
            self._apply_command(payload.get("action"), payload)
            self.lock.notify_all()

    def wait_until_running(self, poll: float = 0.5) -> None:
        """Block while paused; consume one pending step request, if any.

        Called by the host control loop before each step, so the page can
        pause the experiment or let it advance one step at a time.
        """
        with self.lock:
            while (
                self.paused
                and self._step_requests <= 0
                and not self.stop_event.is_set()
            ):
                self.lock.wait(poll)
            if self._step_requests > 0:
                self._step_requests -= 1

    def publish(self, conn: Any) -> None:
        """Capture and publish one frame from the host's connection.

        Called once per host simulation step, so each publish is a new
        frame revision.
        """
        subscribe_vehicles(conn, self._subscribed)
        self._seq += 1
        self._store(capture_frame(conn, self.offset))


# ---------------------------------------------------------------------- #
# HTTP server: static frontend + JSON API + SSE stream
# ---------------------------------------------------------------------- #


def make_viz_handler(
    runtime: Any,
    network_payload: Dict[str, Any],
    viz_dir: Path,
):
    viz_root = viz_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "JevLight3D/1.0"

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            try:
                resolved = path.resolve()
                resolved.relative_to(viz_root)
            except ValueError:
                self._send_json(403, {"error": "forbidden"})
                return
            if not resolved.is_file():
                self._send_json(404, {"error": "not found"})
                return
            body = resolved.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                CONTENT_TYPES.get(resolved.suffix, "application/octet-stream"),
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_file(viz_root / "index.html")
            elif path == "/api/network":
                self._send_json(200, network_payload)
            elif path == "/api/stream":
                self._stream()
            elif path.startswith("/js/") or path.startswith("/css/"):
                self._send_file(viz_root / path.lstrip("/"))
            else:
                self._send_json(404, {"error": f"no route {path}"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/command":
                self._send_json(404, {"error": "no route"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send_json(400, {"error": "bad json"})
                return
            action = payload.get("action")
            if action not in ("play", "pause", "step", "speed", "quit"):
                self._send_json(400, {"error": f"unknown action {action}"})
                return
            if isinstance(runtime, SumoRuntime):
                runtime.command_async(**payload)
            else:
                runtime.command(**payload)
            self._send_json(200, {"ok": True})

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send_event(name: str, data: Any) -> None:
                body = json.dumps(data)
                self.wfile.write(f"event: {name}\ndata: {body}\n\n".encode("utf-8"))
                self.wfile.flush()

            try:
                send_event("init", {
                    "step_length": runtime.step_length,
                    "speed": runtime.speed,
                    "playing": runtime.playing,
                    **network_payload,
                })
                last_seq = -1
                last_send = time.time()
                while not runtime.stop_event.is_set():
                    with runtime.lock:
                        changed = runtime.frame.get("seq") != last_seq
                        if not changed:
                            runtime.lock.wait(timeout=2.0)
                            changed = runtime.frame.get("seq") != last_seq
                        frame = dict(runtime.frame) if changed else None
                    if frame is not None:
                        send_event("frame", frame)
                        last_seq = frame["seq"]
                        last_send = time.time()
                    elif time.time() - last_send > 14.0:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        last_send = time.time()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return  # browser closed the stream

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            print(
                f"[viz] {self.address_string()} {format % args}",
                flush=True,
            )

    return Handler


def make_viz_server(
    runtime: Any,
    network_payload: Dict[str, Any],
    viz_dir: Path = DEFAULT_VIZ_DIR,
    host: str = DEFAULT_VIZ_HOST,
    port: int = DEFAULT_VIZ_PORT,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(
        (host, port),
        make_viz_handler(runtime, network_payload, viz_dir),
    )
    server.daemon_threads = True
    return server


def start_viz_server(
    runtime: Any,
    network_payload: Dict[str, Any],
    viz_dir: Path = DEFAULT_VIZ_DIR,
    host: str = DEFAULT_VIZ_HOST,
    port: int = DEFAULT_VIZ_PORT,
    port_tries: int = 6,
) -> Tuple[ThreadingHTTPServer, str]:
    """Start the viz server on ``port`` (or the next free one) in a thread.

    Returns ``(server, url)``.  On a shared node the default port may be
    taken, so a few consecutive ports are tried before giving up.
    """
    last_error: Optional[OSError] = None
    for candidate in range(port, port + port_tries):
        try:
            server = make_viz_server(
                runtime, network_payload, viz_dir, host, candidate
            )
        except OSError as exc:
            last_error = exc
            continue
        thread = threading.Thread(
            target=server.serve_forever, name="viz-http", daemon=True
        )
        thread.start()
        return server, f"http://{host}:{server.server_address[1]}"
    raise OSError(f"no free viz port in [{port}, {port + port_tries}): {last_error}")


def open_viz_page(url: str) -> None:
    """Best-effort browser open; always print the URL for manual access."""
    print(f"[viz] 3D visualization: {url}", flush=True)
    try:
        import webbrowser

        opened = webbrowser.open(url, new=2)
    except Exception:
        opened = False
    if not opened:
        host = url.split("//", 1)[-1].rsplit(":", 1)[0]
        port = url.rsplit(":", 1)[-1]
        print(
            f"[viz] no local browser (remote/headless session?) — forward the "
            f"port with: ssh -L {port}:{host}:{port} <this-host>, then open "
            f"{url}",
            flush=True,
        )

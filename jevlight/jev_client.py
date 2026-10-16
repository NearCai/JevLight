"""Client for Jev decisions: community MCP, official TypeSafe API, or a local
Jev-compatible server.

Three transports are supported:

- ``mcp`` (default): the Jev community server at
  ``https://www.jevai.org/api/mcp``, called as a stateless MCP streamable-HTTP
  server (JSON-RPC ``tools/call`` on the ``jev_decide`` tool).  Authentication
  is a Bearer ``jev_...`` community API key (``$JEV_API_KEY``).  The endpoint
  sits behind Cloudflare, which rejects default Python user agents, so the
  client sends a curl-style User-Agent.
- ``official``: the TypeSafe System One API (``POST /v1/systemone``) via the
  official ``typesafe-sdk`` when installed, with an equivalent raw-REST
  fallback.  Key: ``$TYPESAFE_API_KEY``.
- ``local``: any self-hosted server that speaks the same System One wire
  format (``POST /v1/systemone``) — e.g. the open Jev-style ecosystem (Laya
  via Ollaya, Jebadiah, Vev, WaterSheep, openjev-sglang, ...) or this repo's
  own ``jevlight.local_server`` adapter over vLLM (see
  ``scripts/serve_jev.sh``).  No API key required; base URL defaults to
  ``$JEV_LOCAL_BASE_URL`` or ``http://127.0.0.1:8123``.

All transports answer the same semantic request — a ``state`` plus a map of
typed ``questions`` (``choice`` / ``noul`` / ``score``) — and are normalized
into the same :class:`JevResult`.
"""

from __future__ import annotations

import json
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_MCP_BASE_URL = "https://www.jevai.org/api/mcp"
DEFAULT_OFFICIAL_BASE_URL = "https://api.typesafe.ai"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8123"
LOCAL_BASE_URL_ENV_VAR = "JEV_LOCAL_BASE_URL"
JEV_TRANSPORTS = ("mcp", "official", "local")
JEV_API_KEY_ENV_VARS = ("JEV_API_KEY", "TYPESAFE_API_KEY")
JEV_DECIDE_TOOL = "jev_decide"
# Cloudflare bans default Python user agents on the community endpoint.
MCP_USER_AGENT = "curl/8.5.0"
RATE_LIMIT_MARKERS = ("too many requests", "rate limit", "try again later")
# Transient upstream failures observed on the community endpoint; retried
# with backoff like rate limits.
TRANSIENT_MARKERS = ("could not be completed", "overloaded", "temporarily")


def resolve_jev_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the Jev API key from an argument or the environment."""
    if explicit:
        return explicit
    for name in JEV_API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


@dataclass
class JevResult:
    """One Jev API call outcome, transport-agnostic."""

    answers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0
    error: Optional[str] = None
    rate_limited: bool = False
    http_status: Optional[int] = None
    credential_id: Optional[str] = None
    credential_attempts: int = 0


class JevClient:
    """Ask typed questions against a state on a Jev decision model."""

    def __init__(
        self,
        transport: str = "mcp",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        min_interval: float = 0.0,
        mock: bool = False,
        batch_size: int = 0,
        key_pool: Optional[str | bool] = None,
    ):
        if transport not in JEV_TRANSPORTS:
            raise ValueError(
                f"transport must be one of {JEV_TRANSPORTS}, got {transport}"
            )
        self.transport = transport
        self.model = model
        # The local transport never resolves keys from the environment: a
        # hosted $JEV_API_KEY must not leak to third-party local servers.
        # An explicitly passed key is still sent (guarded local servers).
        self.api_key = (
            api_key if transport == "local" else resolve_jev_api_key(api_key)
        )
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.min_interval = max(0.0, float(min_interval))
        self.mock = mock
        self.batch_size = int(batch_size)
        if self.batch_size < 0 or (self.batch_size and transport != 'local'):
            raise ValueError('batch_size must be nonnegative and requires local transport')
        self._last_request_time = 0.0
        if transport == "mcp":
            self.base_url = (base_url or DEFAULT_MCP_BASE_URL).rstrip("/")
        elif transport == "local":
            self.base_url = (
                base_url
                or os.environ.get(LOCAL_BASE_URL_ENV_VAR)
                or DEFAULT_LOCAL_BASE_URL
            ).rstrip("/")
        else:
            self.base_url = (base_url or DEFAULT_OFFICIAL_BASE_URL).rstrip("/")
        self._key_pool = None
        self._pool_clients = {}
        self._pool_lock = threading.Lock()
        pool_path = os.environ.get("JEV_API_KEY_POOL") if key_pool is None else key_pool
        if pool_path and transport == "official" and not mock:
            from .api_key_pool import KeyPool
            if self.base_url != DEFAULT_OFFICIAL_BASE_URL:
                raise ValueError("Credential pool is restricted to the official TypeSafe endpoint")
            self._key_pool = KeyPool(pool_path)
            self.api_key = self._key_pool.select()["secret"]
        self._sdk_client = None
        self._mcp_session = None
        self._mcp_initialized = False
        if not mock:
            if self.api_key is None and transport != "local":
                raise ValueError(
                    "Jev API key missing: pass api_key or set one of "
                    + ", ".join(JEV_API_KEY_ENV_VARS)
                )
            if transport == "official" and self._key_pool is None:
                try:
                    from typesafe_sdk import RetryPolicy, TypeSafeClient

                    self._sdk_client = TypeSafeClient(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout,
                        retry=RetryPolicy(max_retries=self.max_retries, timeout=self.timeout),
                    )
                except ImportError:
                    self._sdk_client = None

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def evaluate_batch(self, requests):
        """One round's independent states; no cross-intersection state merging."""
        import requests as http

        if not requests:
            return []
        if not self.batch_size or self.transport != 'local':
            raise ValueError('local batch transport is not enabled')
        if self.mock:
            return [self._mock_evaluate(item['state'], item['questions'], time.time())
                    for item in requests]
        self._pace()
        started = time.time()
        headers = {'Content-Type': 'application/json'}
        if self.api_key is not None:
            headers['Authorization'] = f'Bearer {self.api_key}'
        for attempt in range(self.max_retries + 1):
            try:
                response = http.post(f'{self.base_url}/v1/systemone/batch',
                    json={'requests': requests, 'batch_size': self.batch_size},
                    headers=headers, timeout=self.timeout)
                response.raise_for_status()
                outcomes = response.json()['results']
                if not isinstance(outcomes, list) or len(outcomes) != len(requests):
                    raise ValueError('batch result count mismatch')
                return [JevResult(answers=outcome['answers'], model=outcome.get('model'),
                    input_tokens=int(outcome.get('usage', {}).get('input_tokens', 0)),
                    output_tokens=int(outcome.get('usage', {}).get('output_tokens', 0)),
                    latency=time.time()-started) for outcome in outcomes]
            except Exception as exc:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return [JevResult(error=f'{type(exc).__name__}: {exc}',
                    latency=time.time()-started,
                    rate_limited=self._is_rate_limited(str(exc))) for _ in requests]

    def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Any],
    ) -> JevResult:
        """Evaluate ``questions`` against ``state`` in one Jev request.

        ``latency`` in the result starts after the pacing sleep (which is
        self-imposed waiting, not service time) but includes retry backoff —
        that is the real time the control loop waits for an answer.
        """
        if self.mock:
            return self._mock_evaluate(state, questions, time.time())
        self._pace()
        if self._key_pool is not None:
            def call(key):
                with self._pool_lock:
                    client = self._pool_clients.get(key["id"])
                    if client is None:
                        client = JevClient(transport="official", model=self.model,
                                           api_key=key["secret"], base_url=self.base_url,
                                           timeout=self.timeout, max_retries=self.max_retries,
                                           key_pool=False)
                        self._pool_clients[key["id"]] = client
                return client._evaluate_once(state, questions)
            return self._key_pool.evaluate(call)
        return self._evaluate_once(state, questions)

    def _evaluate_once(self, state, questions):
        started = time.time()
        if self.transport == "mcp":
            return self._evaluate_mcp(state, questions, started)
        if self.transport == "local":
            # Local Jev-compatible server: no key needed, and the ``model``
            # field is only sent when the caller chose one — the server may
            # host a single open-weight checkpoint that picks itself.
            return self._evaluate_systemone_rest(
                state,
                questions,
                started,
                model_field=self.model,
                authorized=self.api_key is not None,
            )
        if self._sdk_client is not None:
            return self._evaluate_sdk(state, questions, started)
        return self._evaluate_systemone_rest(
            state,
            questions,
            started,
            model_field=self.model or DEFAULT_JEV_MODEL,
            authorized=True,
        )

    def _pace(self) -> None:
        """Sleep so consecutive decisions stay at least ``min_interval`` apart."""
        if self.min_interval <= 0:
            return
        elapsed = time.time() - self._last_request_time
        if 0 <= elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------ #
    # Community MCP transport
    # ------------------------------------------------------------------ #

    def _mcp_post(
        self,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """One JSON-RPC POST; returns the parsed ``result`` or raises."""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": MCP_USER_AGENT,
        }
        if self._mcp_session:
            headers["mcp-session-id"] = self._mcp_session
        response = requests.post(
            self.base_url,
            headers=headers,
            json=dict(payload),
            timeout=self.timeout,
        )
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._mcp_session = session_id
        response.raise_for_status()
        body = response.json()
        if body.get("error") is not None:
            raise RuntimeError(f"MCP error: {body['error']}")
        return body.get("result") or {}

    def _mcp_initialize(self) -> None:
        """Best-effort MCP handshake; the community server is stateless."""
        if self._mcp_initialized:
            return
        self._mcp_initialized = True
        try:
            result = self._mcp_post({
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "jevlight", "version": "0.1.0"},
                },
            })
            server = (result.get("serverInfo") or {}).get("name")
            if server:
                print(f"[JevLight] MCP server: {server}", flush=True)
        except Exception as exc:
            print(
                f"[JevLight] MCP initialize skipped ({type(exc).__name__}: {exc})",
                flush=True,
            )

    def _evaluate_mcp(
        self,
        state: Any,
        questions: Mapping[str, Any],
        started: float,
    ) -> JevResult:
        if self._mcp_session is None:
            self._mcp_initialize()
        arguments: Dict[str, Any] = {
            "state": state,
            "questions": dict(questions),
        }
        if self.model:
            arguments["model"] = self.model
        payload = {
            "jsonrpc": "2.0",
            "id": "decide",
            "method": "tools/call",
            "params": {"name": JEV_DECIDE_TOOL, "arguments": arguments},
        }
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                result = self._mcp_post(payload)
                if result.get("isError"):
                    text = self._mcp_result_text(result)
                    if self._should_retry(text) and attempt < self.max_retries:
                        self._pace()
                        time.sleep(delay)
                        delay *= 2
                        continue
                    return JevResult(
                        latency=time.time() - started,
                        error=f"jev_decide error: {text}",
                        rate_limited=self._is_rate_limited(text),
                    )
                return self._parse_decide_result(result, started)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if self._is_rate_limited(message) and attempt < self.max_retries:
                    self._pace()
                    time.sleep(delay)
                    delay *= 2
                    continue
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return JevResult(
                    latency=time.time() - started,
                    error=message,
                    rate_limited=self._is_rate_limited(message),
                )
        return JevResult(
            latency=time.time() - started,
            error="exhausted retries",
            rate_limited=True,
        )

    @staticmethod
    def _mcp_result_text(result: Mapping[str, Any]) -> str:
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                return str(item.get("text", ""))
        return json.dumps(result)

    @staticmethod
    def _is_rate_limited(message: str) -> bool:
        lowered = (message or "").lower()
        return any(marker in lowered for marker in RATE_LIMIT_MARKERS)

    @staticmethod
    def _should_retry(message: str) -> bool:
        """Rate limits and transient upstream failures are worth retrying."""
        lowered = (message or "").lower()
        markers = RATE_LIMIT_MARKERS + TRANSIENT_MARKERS
        return any(marker in lowered for marker in markers)

    def _parse_decide_result(
        self,
        result: Mapping[str, Any],
        started: float,
    ) -> JevResult:
        """Normalize a successful ``jev_decide`` tool result.

        The tool returns its payload as the first text content item.  Accept
        both the full official shape ``{"model", "answers", "usage"}`` and a
        bare ``{question_id: answer}`` map.
        """
        text = self._mcp_result_text(result)
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return JevResult(
                latency=time.time() - started,
                error=f"unparsable jev_decide content: {text[:200]}",
            )
        if not isinstance(data, dict):
            return JevResult(
                latency=time.time() - started,
                error=f"unexpected jev_decide content: {text[:200]}",
            )
        if isinstance(data.get("answers"), dict):
            usage = data.get("usage") or {}
            return JevResult(
                answers=data["answers"],
                model=data.get("model"),
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                latency=time.time() - started,
            )
        # Bare answers map.
        return JevResult(
            answers=data,
            latency=time.time() - started,
        )

    # ------------------------------------------------------------------ #
    # Official TypeSafe transports
    # ------------------------------------------------------------------ #

    def _evaluate_sdk(
        self,
        state: Any,
        questions: Mapping[str, Any],
        started: float,
    ) -> JevResult:
        try:
            # The SDK uses httpx2.Client, which supports sharing across threads.
            # Keep network calls concurrent; the controller bounds worker count.
            response = self._sdk_client.system_one(
                state=state,
                questions=dict(questions),
                model=self.model or DEFAULT_JEV_MODEL,
                timeout=self.timeout,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return JevResult(
                latency=time.time() - started,
                error=message,
                http_status=getattr(exc, "status", getattr(exc, "status_code", None)),
                rate_limited=self._is_rate_limited(message),
            )
        answers: Dict[str, Dict[str, Any]] = {}
        for question_id, answer in (response.answers or {}).items():
            answer_dict: Dict[str, Any] = {"type": answer.type}
            if answer.type == "choice":
                answer_dict["choice"] = answer.choice
                answer_dict["probabilities"] = dict(answer.probabilities)
                answer_dict["confidence"] = float(answer.confidence)
            elif answer.type == "noul":
                answer_dict["noul"] = float(answer.noul)
            elif answer.type == "score":
                answer_dict["score"] = answer.score
                answer_dict["legend"] = dict(answer.legend)
                answer_dict["probabilities"] = dict(answer.probabilities)
                answer_dict["confidence"] = float(answer.confidence)
            answers[question_id] = answer_dict
        usage = response.usage
        return JevResult(
            answers=answers,
            model=getattr(response, "model", None),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            latency=time.time() - started,
        )

    def _evaluate_systemone_rest(
        self,
        state: Any,
        questions: Mapping[str, Any],
        started: float,
        *,
        model_field: Optional[str],
        authorized: bool,
    ) -> JevResult:
        """POST the System One wire format to ``{base_url}/v1/systemone``.

        Shared by the ``official`` transport (always authenticated, model
        defaults to :data:`DEFAULT_JEV_MODEL`) and ``local`` (optional auth,
        model only sent when the caller picked one).
        """
        import requests

        payload: Dict[str, Any] = {
            "state": state,
            "questions": dict(questions),
        }
        if model_field:
            payload["model"] = model_field
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/systemone",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in (429, 529) and attempt < self.max_retries:
                    retry_after = response.headers.get("retry-after")
                    time.sleep(
                        float(retry_after)
                        if retry_after is not None
                        else delay
                    )
                    delay *= 2
                    continue
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage") or {}
                return JevResult(
                    answers=body.get("answers") or {},
                    model=body.get("model"),
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    latency=time.time() - started,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt < self.max_retries and not (self.transport == "official" and status == 402):
                    time.sleep(delay)
                    delay *= 2
                    continue
                return JevResult(
                    latency=time.time() - started,
                    error=message,
                    http_status=status,
                    rate_limited=self._is_rate_limited(message),
                )
        return JevResult(
            latency=time.time() - started,
            error="exhausted retries",
            rate_limited=True,
        )

    # ------------------------------------------------------------------ #
    # Mock transport for pipeline smoke tests
    # ------------------------------------------------------------------ #

    def _mock_evaluate(
        self,
        state: Any,
        questions: Mapping[str, Any],
        started: float,
    ) -> JevResult:
        """Deterministic max-pressure answers computed from the state itself."""
        state_dict = state if isinstance(state, dict) else {}
        intersections = state_dict.get("intersections")
        if intersections is None:
            intersections = {"intersection": state_dict}
        answers: Dict[str, Dict[str, Any]] = {}
        for question_id in questions:
            if not question_id.startswith("phase_"):
                answers[question_id] = {"type": "noul", "noul": 0.25}
                continue
            lanes = intersections.get(question_id[len("phase_"):], {})
            lanes = lanes.get("lanes", {})
            scores = {
                phase: sum(
                    self._mock_lane_pressure(lanes.get(phase[index : index + 2], {}))
                    for index in range(0, len(phase), 2)
                )
                for phase in ("ETWT", "NTST", "ELWL", "NLSL")
            }
            best = max(sorted(scores), key=scores.get)
            answers[question_id] = {
                "type": "choice",
                "choice": best,
                "probabilities": {name: 0.25 for name in scores},
                "confidence": 0.5,
            }
        return JevResult(
            answers=answers,
            model=f"{self.model or 'jev'}-mock",
            input_tokens=1024,
            output_tokens=len(questions) * 8,
            latency=time.time() - started + 0.05,
        )

    @staticmethod
    def _mock_lane_pressure(lane: Mapping[str, Any]) -> float:
        queue = float(lane.get("queue", 0) or 0)
        approaching = sum(lane.get("approaching_by_segment", ()) or [])
        wait = float(lane.get("average_waiting_time_s", 0) or 0)
        return queue + approaching + 0.1 * wait

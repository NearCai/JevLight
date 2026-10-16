"""Local Jev-compatible decision service for open Jev-style models.

Serves the System One wire format (``POST /v1/systemone``: a ``state`` plus
typed ``questions`` in, ``{model, answers, usage}`` out) so that JevLight's
``local`` transport — or any other Jev client — can run against open-weight
decision models on your own hardware instead of the hosted Jev endpoints.

Two backends:

- ``vllm``: any OpenAI-compatible completions endpoint (vLLM, SGLang, …)
  serving an open causal-LM checkpoint such as Tev1 or Laya.  Each question
  becomes one letter-labelled multiple-choice prompt; the option
  probabilities are read from the next-token logprobs in a single prefill —
  the logit-readout approach used by the open System One reproductions.  No
  decoding, milliseconds per question.  Caveat: these are uncalibrated model
  scores, not Jev's calibrated probabilities/confidence.
- ``mock``: deterministic max-pressure answers with the exact semantics of
  ``JevClient(mock=True)``, for pipeline tests without a GPU.

Only the standard library plus ``requests`` is used, matching the project's
dependency set.  Start via ``scripts/serve_jev.sh`` or ``python -m
jevlight.local_server --help``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import requests

from jevlight.jev_client import JevClient


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8123
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8124/v1"
DEFAULT_LOGPROBS = 20
# Logprob penalty for options whose label never appears in the top-k
# next-token logprobs (cannot be scored directly).
MISSING_LABEL_FLOOR = 5.0
QUESTION_WORKERS = 8


# ---------------------------------------------------------------------- #
# Prompt construction and logprob math
# ---------------------------------------------------------------------- #


def render_state(state: Any) -> str:
    """Compact, deterministic JSON rendering of the shared state."""
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def option_letter(index: int) -> str:
    return chr(ord("A") + index)


def choice_prompt(
    state: Any,
    instructions: str,
    criteria: Sequence[Any],
) -> Tuple[str, List[str]]:
    """One letter-labelled multiple-choice prompt for a choice/score question.

    Returns ``(prompt, option_ids)`` — option ``i`` is labelled with the
    ``i``-th letter, and the model answers with a single letter.
    """
    option_ids = list(criteria)
    lines = [
        "You are a fast decision model. Read the STATE, answer the QUESTION "
        "by replying with the single letter of the best option.",
        "",
        "STATE:",
        render_state(state),
        "",
        "QUESTION:",
        str(instructions),
        "",
        "OPTIONS:",
    ]
    for index, option_id in enumerate(option_ids):
        description = criteria[option_id] if isinstance(criteria, Mapping) else None
        if description:
            lines.append(f"{option_letter(index)}. {option_id} — {description}")
        else:
            lines.append(f"{option_letter(index)}. {option_id}")
    lines.append("")
    lines.append("Answer with one letter:")
    return "\n".join(lines), option_ids


def noul_prompt(state: Any, instructions: str) -> str:
    return "\n".join([
        "You are a fast decision model. Read the STATE and answer the "
        "QUESTION with Yes or No.",
        "",
        "STATE:",
        render_state(state),
        "",
        "QUESTION:",
        str(instructions),
        "",
        "Answer Yes or No:",
    ])


def label_variants(label: str) -> Tuple[str, ...]:
    """Token spellings of one answer label across tokenizers."""
    lowered = label.lower()
    return (label, f" {label}", f"{label}.", f" {label}.", lowered, f" {lowered}")


def label_logprob(top_logprobs: Mapping[str, float], label: str) -> Optional[float]:
    """Best logprob the top-k next-token table assigns to ``label``."""
    scores = [
        float(value)
        for variant in label_variants(label)
        for value in [top_logprobs.get(variant)]
        if value is not None
    ]
    return max(scores) if scores else None


def softmax_scores(logprobs: Mapping[str, Optional[float]]) -> Dict[str, float]:
    """Softmax over labels; labels with ``None`` (absent from the top-k
    table) are floored below the weakest scored label."""
    scored = {
        key: value for key, value in logprobs.items() if value is not None
    }
    if not scored:
        raise ValueError("no scored labels")
    floor = min(scored.values()) - MISSING_LABEL_FLOOR
    resolved = {
        key: (value if value is not None else floor)
        for key, value in logprobs.items()
    }
    peak = max(resolved.values())
    weights = {key: math.exp(value - peak) for key, value in resolved.items()}
    total = sum(weights.values()) or 1.0
    return {key: weight / total for key, weight in weights.items()}


# ---------------------------------------------------------------------- #
# Backends
# ---------------------------------------------------------------------- #


class VllmBackend:
    """Score questions against an OpenAI-compatible completions endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_VLLM_BASE_URL,
        model: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        logprobs: int = DEFAULT_LOGPROBS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.logprobs = int(logprobs)
        self.reported_model: Optional[str] = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # -- completions endpoint ------------------------------------------ #

    def _complete(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": self.logprobs,
        }
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage") or {}
                self.total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.total_completion_tokens += int(
                    usage.get("completion_tokens") or 0
                )
                if not self.reported_model and body.get("model"):
                    self.reported_model = body.get("model")
                return body
            except Exception as exc:  # noqa: BLE001 - retried, then raised
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            f"completions endpoint failed: {type(last_error).__name__}: {last_error}"
        )

    def _top_logprobs(self, completion: Mapping[str, Any]) -> Dict[str, float]:
        """The next-token top-k logprob table, dict- or list-shaped."""
        logprobs = (completion.get("logprobs") or {})
        table = (logprobs.get("top_logprobs") or [{}])[0]
        if isinstance(table, Mapping):
            return {str(key): float(value) for key, value in table.items()}
        if isinstance(table, list):  # [{"token": ..., "logprob": ...}, ...]
            return {
                str(item.get("token")): float(item.get("logprob"))
                for item in table
                if isinstance(item, Mapping) and item.get("token") is not None
            }
        return {}

    # -- question scoring ----------------------------------------------- #

    def _answer_choice(
        self,
        state: Any,
        question: Mapping[str, Any],
    ) -> Dict[str, Any]:
        criteria = question.get("criteria") or {}
        if not criteria:
            raise ValueError("choice/score question without criteria")
        prompt, option_ids = choice_prompt(
            state, question.get("instructions", ""), criteria
        )
        completion = self._complete(prompt)["choices"][0]
        table = self._top_logprobs(completion)
        raw = {
            option_id: label_logprob(table, option_letter(index))
            for index, option_id in enumerate(option_ids)
        }
        if not any(value is not None for value in raw.values()):
            # No labelled token in the top-k table: fall back to whatever
            # the model actually generated, if it looks like an option.
            text = str(completion.get("text", "")).strip()
            for index, option_id in enumerate(option_ids):
                if text[:2].strip(".") == option_letter(index):
                    raw[option_id] = 0.0
        if not any(value is not None for value in raw.values()):
            raise ValueError("no option label found in logprobs")
        probabilities = softmax_scores(raw)
        best = max(sorted(probabilities), key=probabilities.get)
        confidence = max(probabilities.values())
        answer: Dict[str, Any] = {
            "type": "choice",
            "choice": best,
            "probabilities": probabilities,
            "confidence": confidence,
        }
        if question.get("type") == "score":
            answer["type"] = "score"
            answer["score"] = best
            answer["legend"] = dict(criteria)
        return answer

    def _answer_noul(
        self,
        state: Any,
        question: Mapping[str, Any],
    ) -> Dict[str, Any]:
        completion = self._complete(
            noul_prompt(state, question.get("instructions", ""))
        )["choices"][0]
        table = self._top_logprobs(completion)
        yes = label_logprob(table, "Yes")
        no = label_logprob(table, "No")
        if yes is None and no is None:
            text = str(completion.get("text", "")).strip().lower()
            if text.startswith("yes"):
                yes = 0.0
            elif text.startswith("no"):
                no = 0.0
            else:
                raise ValueError("no Yes/No token found in logprobs")
        scores = {"yes": yes, "no": no}
        probabilities = softmax_scores(scores)
        return {"type": "noul", "noul": probabilities.get("yes", 0.0)}

    def _answer_one(
        self,
        state: Any,
        question: Mapping[str, Any],
    ) -> Dict[str, Any]:
        question_type = question.get("type")
        if question_type in ("choice", "score"):
            return self._answer_choice(state, question)
        if question_type == "noul":
            return self._answer_noul(state, question)
        raise ValueError(f"unsupported question type: {question_type}")

    # -- backend interface ----------------------------------------------- #

    def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=QUESTION_WORKERS) as executor:
            answers = list(
                executor.map(
                    lambda item: self._answer_one(state, item[1]),
                    questions.items(),
                )
            )
        return {
            "model": self.reported_model or self.model or "vllm",
            "answers": {
                question_id: answer
                for question_id, answer in zip(questions, answers)
            },
            "usage": {
                "input_tokens": self.total_prompt_tokens,
                "output_tokens": self.total_completion_tokens,
            },
        }


class MockBackend:
    """Deterministic max-pressure answers; same semantics as ``--mock_jev``."""

    def __init__(self, model: Optional[str] = None):
        self._client = JevClient(transport="local", mock=True, model=model)

    def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        result = self._client.evaluate(state, questions)
        return {
            "model": result.model or "jev-mock",
            "answers": result.answers,
            "usage": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        }


# ---------------------------------------------------------------------- #
# HTTP server
# ---------------------------------------------------------------------- #


def make_handler(backend: Any, model_name: Optional[str]):
    class SystemOneHandler(BaseHTTPRequestHandler):
        server_version = "JevLightLocal/1.0"

        def _send(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _served_model(self) -> str:
            return model_name or getattr(backend, "reported_model", None) or "local"

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            if self.path in ("/health", "/healthz"):
                self._send(200, {"status": "ok", "model": self._served_model()})
            elif self.path == "/v1/models":
                self._send(200, {
                    "object": "list",
                    "data": [{"id": self._served_model(), "object": "model"}],
                })
            else:
                self._send(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            if self.path.rstrip("/") not in ("/v1/systemone", "/v1/systemone/batch"):
                self._send(404, {"error": {"message": f"no route {self.path}"}})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": {"message": f"bad request body: {exc}"}})
                return
            if self.path.rstrip('/') == '/v1/systemone/batch':
                items = request.get('requests')
                size = request.get('batch_size')
                if (not isinstance(items, list) or not items or
                    type(size) is not int or size < 1 or any(
                        not isinstance(item, dict) or
                        not isinstance(item.get('questions'), Mapping) for item in items)):
                    self._send(400, {'error': {'message': 'invalid batch requests or batch_size'}})
                    return
                if not hasattr(backend, 'evaluate_batch'):
                    self._send(400, {'error': {'message': 'backend does not support batches'}})
                    return
                try:
                    outcomes = backend.evaluate_batch(items, size)
                    if len(outcomes) != len(items):
                        raise ValueError('batch result count mismatch')
                    self._send(200, {'results': outcomes})
                except Exception as exc:
                    self._send(502, {'error': {'message': f'{type(exc).__name__}: {exc}'}})
                return
            state = request.get("state")
            questions = request.get("questions")
            if not isinstance(questions, Mapping):
                self._send(400, {
                    "error": {"message": "request must include a questions map"},
                })
                return
            started = time.time()
            try:
                outcome = backend.evaluate(state, questions)
            except Exception as exc:  # noqa: BLE001 - reported to the client
                self._send(502, {
                    "error": {
                        "message": f"{type(exc).__name__}: {exc}",
                        "type": "backend_error",
                    }
                })
                return
            self._send(200, {
                "model": model_name or outcome.get("model") or "local",
                "answers": outcome.get("answers") or {},
                "usage": outcome.get("usage")
                or {"input_tokens": 0, "output_tokens": 0},
                "latency_s": round(time.time() - started, 4),
            })

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            print(
                f"[jev-local] {self.address_string()} {format % args}",
                flush=True,
            )

    return SystemOneHandler


def make_server(
    backend: Any,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model_name: Optional[str] = None,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(backend, model_name))
    server.daemon_threads = True
    return server


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a local Jev-compatible decision model on POST /v1/systemone"
        )
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "mock"),
        default="vllm",
        help=(
            "vllm: score questions on an OpenAI-compatible completions "
            "endpoint; mock: deterministic max-pressure answers (no GPU)"
        ),
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--model",
        default=None,
        help="Model name reported to clients (default: backend's own)",
    )
    parser.add_argument(
        "--vllm_base_url",
        default=DEFAULT_VLLM_BASE_URL,
        help=f"OpenAI-compatible base URL (default: {DEFAULT_VLLM_BASE_URL})",
    )
    parser.add_argument(
        "--vllm_model",
        default=None,
        help="Model id sent to the completions endpoint (default: --model)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=DEFAULT_LOGPROBS,
        help="Top-k next-token logprobs requested per question",
    )
    args = parser.parse_args(argv)

    if args.backend == "mock":
        backend = MockBackend(model=args.model)
    else:
        backend = VllmBackend(
            base_url=args.vllm_base_url,
            model=args.vllm_model or args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            logprobs=args.logprobs,
        )
    server = make_server(backend, args.host, args.port, args.model)
    print(
        f"[jev-local] backend={args.backend} "
        f"listening on http://{args.host}:{args.port}/v1/systemone",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

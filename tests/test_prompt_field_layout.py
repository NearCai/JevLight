"""Shared policy must not consume each native candidate's small text budget."""
from types import SimpleNamespace
from types import ModuleType
import sys

import pytest

from jevlight.controller import (
    PROMPT_VERSION, cojevlight_phase_question, jevlight_phase_question,
)


def test_shared_policy_is_present_once_and_movement_meanings_stay_in_candidates():
    observation = SimpleNamespace(phases=["WT_ET", "NT_ST", "WL_EL", "NL_SL"])
    local = jevlight_phase_question(observation, 15)
    co = cojevlight_phase_question(observation, 15, "$")
    assert local["instructions"].count("largest combined queue, approaching, and waiting-time pressure") == 1
    assert local["criteria"] == co["criteria"]
    for phase, description in local["criteria"].items():
        assert phase[:2] in description and phase[2:] in description
        assert "arriving from" in description
        assert "pressure" not in description
        assert "Prefer" not in description
    assert "shared-instructions" in PROMPT_VERSION


@pytest.mark.parametrize("tokens", [4096, 4097])
def test_openjev_counts_token_ids_not_batchencoding_fields(monkeypatch, tokens):
    from jevlight.backends.decision_model_service import OpenJevNative
    package, api = ModuleType("jev"), ModuleType("jev.api")
    api.compile_request = lambda **request: [request]
    api.candidate_prompts = lambda record: ["entire candidate path"]
    monkeypatch.setitem(sys.modules, "jev", package)
    monkeypatch.setitem(sys.modules, "jev.api", api)

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            if tokenize:
                return {"input_ids": [1] * tokens, "attention_mask": [1] * tokens}
            return "rendered full prompt"

        def __call__(self, text, *, truncation):
            assert text == "rendered full prompt" and truncation is False
            return {"input_ids": [1] * tokens}

    runtime = OpenJevNative.__new__(OpenJevNative)
    runtime.model = SimpleNamespace(tokenizer=Tokenizer(), max_length=4096)
    if tokens > 4096:
        with pytest.raises(ValueError, match="4097 tokens exceeds limit 4096"):
            runtime.encode({})
    else:
        assert runtime.encode({})[0]["candidate_path_tokens"] == [4096]


def test_candidate_embedding_limit_has_distinct_context_details():
    from jevlight.backends.context_limits import context_details, is_context_limit
    reason = "phase_x: native candidate embedding cap 64 exceeded: [12, 70]"
    assert is_context_limit(reason)
    details = context_details(reason)
    assert details["native_option_limit_tokens"] == 64
    assert details["actual_option_max_tokens"] == 70

# JevLight

JevLight is a [CityFlow](https://github.com/cityflow-project/CityFlow)-based
traffic-signal-control framework. It evaluates Jev and OpenAI-compatible LLM
controllers alongside rule-based and reinforcement-learning baselines through
the same simulation pipeline.

## Features

- Structured Jev decisions for signal phase and green-time duration.
- Explicit traffic-state prompts with queue, waiting-time, movement, and phase
  urgency information.
- CityFlow runners for Jinan, Hangzhou, and New York scenarios.
- Reproducible rule-based, RL, and ChatGPT-compatible baselines.
- Structured decision records for prompt/reply inspection.

## Requirements

- Python 3.9 or a compatible Python environment
- CityFlow
- NumPy, pandas, PyTorch, TensorFlow, tqdm, W&B, and requests
- A Linux environment is recommended for CityFlow experiments

Install the runtime dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install cityflow numpy pandas torch tensorflow tqdm wandb requests
```

## Quick start

### Dataset layout

Place the road network and traffic-flow files under the matching dataset
directory:

```text
data/
|-- Jinan/
`-- 3_4/
    |-- roadnet_3_4.json
    `-- anon_3_4_jinan_real.json
```

Supported dataset names are:

| `--dataset` | Directory | Road network |
| --- | --- | --- |
| `jinan` | `data/Jinan/3_4/` | `roadnet_3_4.json` |
| `hangzhou` | `data/Hangzhou/4_4/` | `roadnet_4_4.json` |
| `newyork_28x7` | `data/NewYork/28_7/` | `roadnet_28_7.json` |

### Run JevLight

Set the API key locally; `.env.example` intentionally contains no credential.

```bash
export TYPESAFE_API_KEY="<your-api-key>"
python run_jev.py \
  --dataset jinan \
  --traffic_file anon_3_4_jinan_real.json \
  --phase_mode choice \
  --duration_mode choice
```

The equivalent PowerShell setup is:

```powershell
$env:TYPESAFE_API_KEY = "<your-api-key>"
python run_jev.py --dataset jinan --traffic_file anon_3_4_jinan_real.json
```

`run_jev.py` also accepts `--api_key`, `--endpoint`, `--jev_model`,
`--timeout`, `--run_counts`, and `--max_concurrency`. API failures can use the
local safety fallback; pass `--no_fallback` when a live API response is
required.

### Run an OpenAI-compatible baseline

For SiliconFlow/Qwen:

```bash
export CHATGPT_API_KEY="<your-api-key>"
python run_chatgpt.py --prompt Commonsense \
  --api_base https://api.siliconflow.cn/v1 \
  --gpt_version Qwen/Qwen2.5-7B-Instruct
```

For Ollama, use an installed model and
`--api_base http://localhost:11434/v1`, for example:

```bash
python run_chatgpt.py --prompt Commonsense \
  --api_base http://localhost:11434/v1 \
  --gpt_version qwen2.5:7b
```

Use `--no_fallback` to require a live model endpoint. `--run_counts` and
`--timeout` control the simulation horizon and request timeout.

### Run checks

```bash
python run_jev.py --help
python -m unittest discover -s tests -v
```

## Decision modes

Both `--phase_mode` and `--duration_mode` support the following modes:

| Mode | Description |
| --- | --- |
| `choice` | Select one item from the supplied discrete options. |
| `score` | Rank the options using returned level scores/probabilities. |
| `noul` | Use the corresponding Jev mode without discrete-choice projection. |

The recommended configuration is `choice` for both phase and duration. Jev
chooses one of four phases and a green duration from `15`, `20`, `25`, `30`,
`35`, or `40` seconds. The simulator checks for decisions every five seconds;
changing phase inserts a separate five-second yellow transition.

## Baselines and reproduction

| Category | Entrypoints |
| --- | --- |
| Rule-based | `run_random.py`, `run_fixedtime.py`, `run_maxpressure.py` |
| Reinforcement learning | `run_presslight.py`, `run_mplight.py`, `run_colight.py`, `run_dynamiclight.py`, `run_rl_eval.py` |
| LLM-based | `run_chatgpt.py`, `run_jev.py` |

To reproduce the full-horizon RL rows on Jinan:

```bash
for model in PressLight MPLight Colight DynamicLight; do
  WANDB_MODE=offline python run_rl_eval.py --model "$model" --run_counts 3600
done
```

All runners use the shared CityFlow configuration and environment. For
publication-quality comparisons, use the same seed, horizon, and repeated
trials for every controller.

## Benchmark

The snapshot below uses `Jinan/3_4/anon_3_4_jinan_real.json` and a 3,600-second
horizon. Queue length, waiting time, and travel time are emitted by `OneLine`;
lower values are better. The machine-readable copy is
[`results/benchmark_jinan.json`](results/benchmark_jinan.json).

| Controller | Horizon (s) | Avg queue | Avg waiting time (s) | Avg travel time (s) |
| --- | ---: | ---: | ---: | ---: |
| Random | 3,600 | 630.59 | 35.55 | 594.16 |
| Fixedtime (30 s) | 3,600 | 431.37 | 50.70 | 451.45 |
| MaxPressure | 3,600 | 199.68 | 30.76 | 317.51 |
| PressLight | 3,600 | 697.76 | 41.33 | 630.93 |
| MPLight | 3,600 | 391.74 | 26.49 | 439.62 |
| CoLight | 3,600 | 866.49 | 51.12 | 750.91 |
| DynamicLight | 3,600 | 609.80 | 92.07 | 568.30 |
| Qwen2.5-7B-Instruct | 3,600 | 189.14 | 25.34 | 312.47 |
| Jev (API, revised prompt) | 3,600 | 206.20 | 47.76 | 312.75 |

The Jev API run recorded 978 successful decisions, 95 transient fallbacks, and
4 backlog-guardrail corrections. All 95 saved `decision_error` entries in that
run were `HTTPSConnectionPool(...): Read timed out (read timeout=2.0)`, so they
were request read timeouts rather than invalid traffic decisions. The current
runner's default timeout is 30 seconds; use a live endpoint and matched seeds
when comparing new prompt or model versions.

## Project layout

```text
models/      Traffic-control agents, including the Jev adapter
utils/       CityFlow environment, configuration, and shared pipelines
prompts/     Prompt templates for the ChatGPT-compatible agent
tests/       Jev contract and scheduling checks
run_*.py     Experiment entrypoints
results/     Versioned benchmark summaries
```

## Configuration

The Jev adapter follows the
[TypeSafe Jev quickstart](https://docs.typesafe.ai/introduction/quickstart).
The default endpoint is `https://api.typesafe.ai/v1/systemone`; replace it with
the endpoint and model supplied by your deployment when needed. Credentials are
read from command-line arguments or environment variables and are not stored
in the repository.

## Acknowledgements

We gratefully acknowledge the authors and contributors of
[LLMTSCS](https://github.com/usail-hkust/LLMTSCS), whose traffic-signal-control
implementation provided an important foundation for this project, and
[CityFlow](https://github.com/cityflow-project/CityFlow), whose high-performance
traffic simulator powers the experiments and benchmarks.

## License

JevLight is released under the [MIT License](LICENSE).

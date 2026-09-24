# JevLight

JevLight is a [CityFlow](https://github.com/cityflow-project/CityFlow) traffic
signal control framework. It provides shared runners for Jev, Laya,
rule-based controllers, and reinforcement-learning baselines.

## Features

- Structured phase and green-time decisions.
- Jinan, Hangzhou, and New York CityFlow scenarios.
- Reproducible simulation records and benchmark summaries.
- API and fully local decision-model runners.

## Requirements

- Python 3.10+
- CityFlow, NumPy, pandas, PyTorch, TensorFlow, tqdm, W&B, and requests
- `laya==0.3.7` for the local Laya runner

```bash
python -m venv .venv
source .venv/bin/activate
pip install cityflow numpy pandas torch tensorflow tqdm wandb requests laya==0.3.7
```

## Quick start

Datasets are included under `data/`:

| Dataset | Directory | Road network |
| --- | --- | --- |
| `jinan` | `data/Jinan/3_4/` | `roadnet_3_4.json` |
| `hangzhou` | `data/Hangzhou/4_4/` | `roadnet_4_4.json` |
| `newyork_28x7` | `data/NewYork/28_7/` | `roadnet_28_7.json` |

### Jev API

Set `TYPESAFE_API_KEY` locally, then run:

```bash
export TYPESAFE_API_KEY="<your-api-key>"
python run_jev.py \
  --dataset jinan \
  --traffic_file anon_3_4_jinan_real.json \
  --phase_mode choice \
  --duration_mode choice
```

Use `--no_fallback` to require a live API response. The runner also accepts
`--endpoint`, `--jev_model`, `--timeout`, `--run_counts`, and
`--max_concurrency`.

### Local Laya

The local runner uses `convaiinnovations/laya-typed-decisions` for four phase
choices and six green-time choices (`15`--`40` seconds).

```bash
export HF_HOME="$PWD/.cache/huggingface"
WANDB_MODE=offline python run_laya.py \
  --dataset jinan \
  --traffic_file anon_3_4_jinan_real.json \
  --run_counts 3600 \
  --device cuda
```

Use `--device cpu` without CUDA. The first run downloads the checkpoint to the
local Hugging Face cache, which is ignored by Git. Use `--no_guardrail` to
disable the safety rule or `--min_confidence` to enable low-confidence
fallbacks. Decision probabilities are saved with the simulation trace.

### OpenAI-compatible baseline

```bash
export CHATGPT_API_KEY="<your-api-key>"
python run_chatgpt.py --prompt Commonsense \
  --api_base https://api.siliconflow.cn/v1 \
  --gpt_version Qwen/Qwen2.5-7B-Instruct
```

### Checks

```bash
python run_jev.py --help
python -m unittest discover -s tests -v
```

## Decision modes

Jev supports `choice`, `score`, and `noul` modes for both phase and duration
decisions. `choice` is the recommended mode. The simulator evaluates actions
every five seconds and applies a five-second yellow transition when the phase
changes.

## Baselines

| Category | Entrypoints |
| --- | --- |
| Rule-based | `run_random.py`, `run_fixedtime.py`, `run_maxpressure.py` |
| Reinforcement learning | `run_presslight.py`, `run_mplight.py`, `run_colight.py`, `run_dynamiclight.py`, `run_rl_eval.py` |
| Decision models | `run_chatgpt.py`, `run_jev.py`, `run_laya.py` |

For a full-horizon RL run on Jinan:

```bash
WANDB_MODE=offline python run_rl_eval.py --model PressLight --run_counts 3600
```

## Benchmark

Jinan 3×4, 3,600-second horizon; lower is better. The complete machine-readable
benchmark is [`results/benchmark_jinan.json`](results/benchmark_jinan.json).

| Controller | Avg queue length | Avg waiting time (s) | Avg travel time (s) |
| --- | ---: | ---: | ---: |
| Random | 630.59 | 35.55 | 594.16 |
| Fixedtime (30 s) | 431.37 | 50.70 | 451.45 |
| MaxPressure | 199.68 | 30.76 | 317.51 |
| PressLight | 697.76 | 41.33 | 630.93 |
| MPLight | 391.74 | 26.49 | 439.62 |
| CoLight | 866.49 | 51.12 | 750.91 |
| DynamicLight | 609.80 | 92.07 | 568.30 |
| Qwen2.5-7B-Instruct | 189.14 | 25.34 | 312.47 |
| Jev | 206.20 | 47.76 | 312.75 |
| Laya | 191.39 | 47.36 | 303.87 |

## Project layout

```text
models/      Traffic-control agents
utils/       CityFlow environment and shared pipeline
prompts/     Prompt templates
tests/       Unit and contract tests
run_*.py     Experiment entrypoints
results/     Benchmark summaries
```

## Acknowledgements

We gratefully acknowledge the authors and contributors of
[LLMTSCS](https://github.com/usail-hkust/LLMTSCS), whose traffic-signal-control
implementation provided an important foundation for this project, and
[CityFlow](https://github.com/cityflow-project/CityFlow), whose high-performance
traffic simulator powers the experiments and benchmarks. We also thank the
[Reinforcement Learning for Traffic Signal Control – Open Datasets](https://traffic-signal-control.github.io/#open-datasets)
project for providing the road networks and traffic-flow datasets included in
this repository.

## License

JevLight is released under the [MIT License](LICENSE).

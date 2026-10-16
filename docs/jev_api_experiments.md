# Official Jev API experiments

The API matrix uses the [common experiment settings](experiment_settings.md): seven flows, local/cooperative modes and three seeds, for 42 runs. Each run uses `run_jevlight.py`. The official API's returned valid `choice` controls the signal; probability fields are recorded without normalization or argmax gates. Question IDs, answer types, legal phases and model identity remain checked. Existing network retry and fallback rules apply.

```bash
uv run python scripts/run_jev_api_experiments.py --stage all
```

This performs short SUMO preflights before the formal matrix. Use `--resume <batch-directory> --stage all` to resume compatible completed conditions; failed or unfinished attempts are retained. Outputs include redacted configuration, original summaries, decisions, request audits and AQL/AWT/ATT metrics under `results/jev-api/`. Baselines run independently.

## Credentials and networking

Provide a single API key through the environment, or configure the optional private key pool with `JEV_API_KEY_POOL`. Vault files remain outside version control and are readable only by their owner. Standard proxy environment variables can be supplied to the process when needed.

The pool keeps a separate client per key and an anonymous usage ledger. An HTTP 402 exhausts the key used by that request and advances to the next; other errors retain the existing retry policy. Exhausting the pool stops the run. Usage records are not an account balance, and unknown balances remain unknown. API keys are sent only to the official API origin.

```bash
uv run python -m jevlight.api_key_pool status
```

# PV+BESS RL Trading 

## Executive Summary

This project teaches an AI to operate a solar + battery plant in the Hungarian electricity markets. The AI learns, by simulation, how to charge/discharge the battery and commit reserves to maximize daily profit while strictly obeying technical limits and accounting for battery wear.

## What’s Inside (map)

- Data & Config
  - `notebooks/` (data preparation, exploration)
  - `config/features.yaml`, `config/limits.yaml`
- Simulator (practice field)
  - `src/envs/env_pv_bess.py`
- AI (learning)
  - `src/rl/models.py` (neural networks: Actor/Critics with LSTM)
  - `src/rl/buffer.py` (experience memory)
  - `src/rl/td3.py` (TD3 learning algorithm)
- Orchestration
  - `scripts/train_td3.py` (training driver)
  - `scripts/eval_baselines.py` (compare to simple traders)
- Sanity
  - `tests/tests_sanity.py`, `tests/test_env_basic.py`

## Markets & Objective

- Day‑ahead energy (HUPX): arbitrage across the day
- Ancillary services/reserves (MAVIR): capacity revenues
- Objective: maximize net daily revenue (revenues − degradation), under all limits

## The Simulator — `src/envs/env_pv_bess.py`

- Acts as a realistic day: 24 decision steps (hours)
- State (inputs): 21 values (time, prices, PV availability, daily context, 4 future-aware signals, discharge-hour flag), normalized to [-1, 1]
- Action (decision): a single continuous control `δ ∈ [-1, 1]` (negative=charge, positive=discharge, 0=idle)
- Enforces only what’s in `config/limits.yaml`: SOC bounds, converter/inverter caps, POI export/import caps, energy capacity, reserve feasibility
- Economics: energy and reserve revenues, plus a non‑linear, DOD‑based degradation cost
- Reward: net cash flow per hour
- No heuristics: the AI is not guided by rules; it must learn

## The Brain — `src/rl/models.py`

- Actor network: LSTM(64) → MLP(64, 64) → outputs `δ` in [-1, 1]
- Twin Critic networks: LSTM+MLP score how good a state+action is (Q‑value)
- Sequence‑first I/O: models read sequences [batch, time, features] to “remember” earlier hours


- A short story: how the “brain” thinks

- First, the notetaker (LSTM, size 64)

Imagine a trader who keeps a running diary through the day. At 08:00 they remember dawn prices, PV ramp-up; by 18:00 they still recall mid‑day conditions. The LSTM is that notetaker: it reads each hour in sequence and decides what to keep or forget. A hidden size of 64 is like giving the trader a compact but capable memory—enough to retain the daily rhythm without overpacking their head.

- Then, the decider (MLP, layers 64 → 64)

After the diary is updated, the trader turns notes into a concrete move: charge, hold, or sell. The MLP is that decision desk. Two modest layers (64, 64) give just enough reasoning depth to capture “if this and that, then do X” patterns, while staying fast and stable. It’s the right-sized committee—large enough to be smart, small enough to stay decisive.

## Memory — `src/rl/buffer.py`

- Replay buffer stores many past experiences (state, action, reward, next_state, done)
- Random sampling breaks short‑term correlations and stabilizes learning
- Optional frame stacking to include short recent history without changing the model

## Learning (TD3) — `src/rl/td3.py`

- TD3 = Twin Delayed Deep Deterministic Policy Gradient
- Stability features:
  - Twin critics (reduce over‑optimistic estimates)
  - Delayed actor updates (fewer, higher‑quality policy updates)
  - Target policy smoothing (small noise for robustness)
  - Soft target updates (Polyak averaging)
- Defaults tuned for this domain: `gamma=1.0`, `lr=5e-5`, `tau=0.01`, `policy_delay=2`, `exploration_std=0.5`, `target_std=0.5`, `batch_size=64` (actions are clamped to [0, 1])

## Training Driver — `scripts/train_td3.py`

- Reproducible: consistent day selection from seeds (NumPy/Torch/Env)
- Warmup with random actions; then learn from the replay buffer
- Safe I/O: immediate CSV flush, clean interrupt handling, checkpoints with a minimal `checkpoint_meta.json`
- Logs per‑episode profit, revenue components, degradation, SOC min/max

## Baselines — `src/baselines/policies.py`

- PV+DS3 (reserve‑only)
- PV+DA (simple two‑peak arbitrage)
- Realistic Trader (Linear Deg): human‑style threshold strategy, ends near 50% SOC
- Purpose: quick benchmarks to gauge AI uplift

## Sanity Tests — `tests/`

- Check SOC bounds, POI caps, and 24‑step episodes under random actions
- Ensure the simulator is safe for learning (not a performance test)

## Configuration You Can Edit

- `config/limits.yaml` — physical/operational limits (SOC, efficiencies, converter/inverter, POI caps, capacity)
- `config/features.yaml` — which inputs feed the AI and how they’re scaled

## Tuning (non‑technical)

- More stability: reduce `exploration_std`, lower `lr`, or increase buffer size
- Faster learning: more epochs and varied training days
- More “memory”: increase LSTM hidden size in `src/rl/models.py`
- Stricter operations: update `config/limits.yaml`

## FAQ (short)

- Does it use rules? No — the simulator simulates; the AI learns.
- Why LSTM? To use context from earlier hours (e.g., charge before peaks).
- Will it over‑cycle the battery? Degradation cost and hard limits prevent this.

## References

- Research basis: `battery_algo_article.pdf`
- HUPX Market: https://hupx.hu/
- MAVIR (Hungarian TSO): https://www.mavir.hu/

## License & Contact

- MIT License — see `LICENSE`
- Questions or collaboration: open an issue

—

Status: active development

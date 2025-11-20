"""Rule-based baseline policies for the BESSEnv environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.envs import BESSEnv


@dataclass
class EpisodeStats:
    """Aggregated outcome metrics for a 24-hour episode."""

    episode_return: float
    episode_profit: float
    episode_revenue: float
    episode_degradation: float
    revenue_pv_grid: float
    revenue_energy: float
    revenue_reserve: float
    cost_degradation: float
    final_soc: float


def _denormalized_prices(env: BESSEnv) -> np.ndarray:
    day_data = env.day_data.reset_index(drop=True)
    return np.array(
        [env._denormalize("price_em", row["price_em"]) for _, row in day_data.iterrows()],
        dtype=np.float64,
    )


def _select_day_from_seed(env: BESSEnv, seed: int) -> str:
    rng = np.random.default_rng(seed)
    return env.dates[int(rng.integers(0, len(env.dates)))]


def _run_plan(env: BESSEnv, plan: List[float]) -> EpisodeStats:
    totals = {
        "return": 0.0,
        "revenue_pv_grid": 0.0,
        "revenue_energy": 0.0,
        "revenue_reserve": 0.0,
        "cost_degradation": 0.0,
    }
    final_info: Dict[str, float] = {}
    for hour, delta in enumerate(plan):
        action = np.array([float(np.clip(delta, -1.0, 1.0))], dtype=np.float32)
        _, reward, terminated, truncated, info = env.step(action)
        totals["return"] += reward
        totals["revenue_pv_grid"] += info.get("revenue_pv_grid", 0.0)
        totals["revenue_energy"] += info.get("revenue_energy", 0.0)
        totals["revenue_reserve"] += info.get("revenue_reserve", 0.0)
        totals["cost_degradation"] += info.get("cost_degradation", 0.0)
        final_info = info
        if terminated or truncated:
            break

    return EpisodeStats(
        episode_return=totals["return"],
        episode_profit=final_info.get("episode_profit", totals["return"]),
        episode_revenue=final_info.get(
            "episode_revenue",
            totals["revenue_pv_grid"] + totals["revenue_energy"] + totals["revenue_reserve"],
        ),
        episode_degradation=final_info.get("episode_degradation", totals["cost_degradation"]),
        revenue_pv_grid=totals["revenue_pv_grid"],
        revenue_energy=totals["revenue_energy"],
        revenue_reserve=totals["revenue_reserve"],
        cost_degradation=totals["cost_degradation"],
        final_soc=final_info.get("soc", env.soc),
    )


def pv_ds3_plan(env: BESSEnv, seed: int) -> List[float]:
    del seed
    return [0.0] * env.max_hours


def pv_da_plan(env: BESSEnv, seed: int) -> List[float]:
    """Light heuristic: small discharge at peaks, modest charging beforehand."""

    del seed
    day_data = env.day_data.reset_index(drop=True)
    plan = [0.0] * env.max_hours
    if day_data.empty:
        return plan

    row0 = day_data.iloc[0]
    morning_peak = int(round(env._denormalize("k_em_max_morning", row0["k_em_max_morning"])))
    evening_peak = int(round(env._denormalize("k_em_max_evening", row0["k_em_max_evening"])))
    prices = _denormalized_prices(env)

    def _clip_hour(h: int) -> int:
        return int(np.clip(h, 0, env.max_hours - 1))

    morning_peak = _clip_hour(morning_peak)
    evening_peak = _clip_hour(evening_peak)

    for peak in sorted({morning_peak, evening_peak}):
        plan[peak] = 0.95  # Discharge at peak hours (positive)
        window = [h for h in range(max(0, peak - 6), peak) if plan[h] == 0.0]
        if window:
            buy_hour = min(window, key=lambda h: prices[h])
            plan[buy_hour] = -0.9  # Charge before peak hours (negative)

    return plan


def pv_da_full_plan(env: BESSEnv, seed: int) -> List[float]:
    """Aggressive paper-style arbitrage: alternating charge/discharge pattern."""

    del seed
    # Alternate between charging and discharging (not practical but for testing)
    return [1.0 if h % 2 == 0 else -1.0 for h in range(env.max_hours)]


def evaluate_policy(env: BESSEnv, name: str, plan_fn, seeds: List[int]) -> Dict[str, float]:
    stats: List[EpisodeStats] = []
    for seed in seeds:
        day = _select_day_from_seed(env, seed)
        env.reset(seed=seed, options={"date": day})
        plan = plan_fn(env, seed)
        stats.append(_run_plan(env, plan))

    mean_profit = float(np.mean([s.episode_profit for s in stats]))
    mean_revenue = float(np.mean([s.episode_revenue for s in stats]))
    mean_deg = float(np.mean([s.episode_degradation for s in stats]))
    mean_soc = float(np.mean([s.final_soc for s in stats]))

    print(
        f"Policy {name}: mean profit {mean_profit:.2f} EUR | revenue {mean_revenue:.2f} EUR | "
        f"degradation {mean_deg:.2f} EUR | final SOC {mean_soc:.3f}"
    )

    return {
        "mean_profit": mean_profit,
        "mean_revenue": mean_revenue,
        "mean_degradation": mean_deg,
        "mean_final_soc": mean_soc,
    }


__all__ = [
    "EpisodeStats",
    "evaluate_policy",
    "pv_ds3_plan",
    "pv_da_plan",
    "pv_da_full_plan",
]



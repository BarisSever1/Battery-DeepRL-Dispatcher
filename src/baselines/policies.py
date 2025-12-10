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
    for hour, action_value in enumerate(plan):
        # Plan is already in [0, 1] format
        action = np.array([float(np.clip(action_value, 0.0, 1.0))], dtype=np.float32)
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


def pv_da_two_cycles_plan(env: BESSEnv, seed: int) -> List[float]:
    """
    Two full cycles baseline using [0, 1] action space:
    1. Start at 0.5 SOC
    2. Charge fully in cheapest hours before morning peak (action=1 outside peak)
    3. Discharge fully during morning peak window (peak-1 to peak+1, action=1 in peak)
    4. Charge before evening peak in cheap hours (action=1 outside peak)
    5. Discharge fully during evening peak window (peak-1 to peak+1, action=1 in peak)
    Note: Does not charge back to 0.5 - ends at whatever SOC after evening discharge
    
    Returns actions in [0, 1] format where:
    - In peak windows: 1 = discharge, 0 = idle
    - Outside peak windows: 1 = charge, 0 = idle
    """
    del seed
    day_data = env.day_data.reset_index(drop=True)
    plan = [0.0] * env.max_hours  # Default: idle (0.0)
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

    # Use environment's peak windows (set during reset) to ensure consistency
    morning_peak_hours = getattr(env, 'morning_peak_hours', set())
    evening_peak_hours = getattr(env, 'evening_peak_hours', set())
    all_peak_hours = morning_peak_hours | evening_peak_hours
    
    # Also keep the window sets for discharge logic
    morning_peak_window = morning_peak_hours
    evening_peak_window = evening_peak_hours

    # Find cheapest hours before morning peak (hours 0 to morning_peak-1)
    hours_before_morning = list(range(0, morning_peak))
    cheapest_before_morning = []
    if hours_before_morning:
        prices_before_morning = [(h, prices[h]) for h in hours_before_morning]
        prices_before_morning.sort(key=lambda x: x[1])  # Sort by price
        # Take 3 cheapest hours before morning peak
        cheapest_before_morning = [h for h, _ in prices_before_morning[:3]]

    # Find cheapest hours before evening peak (between morning and evening peak)
    hours_before_evening = list(range(morning_peak + 1, evening_peak))
    cheapest_before_evening = []
    if hours_before_evening:
        prices_before_evening = [(h, prices[h]) for h in hours_before_evening]
        prices_before_evening.sort(key=lambda x: x[1])  # Sort by price
        # Take 3 cheapest hours before evening peak
        cheapest_before_evening = [h for h, _ in prices_before_evening[:3]]

    # Phase 1: Charge fully in cheapest hours before morning peak
    # Action = 1.0 if outside peak window (charge), 0.0 if in peak window
    for h in cheapest_before_morning:
        # Always charge in cheapest hours (they should be outside peak windows)
        plan[h] = 1.0  # Charge (action=1 outside peak)

    # Phase 2: Discharge fully during morning peak window (peak-1 to peak+1)
    # Action = 1.0 if in peak window (discharge), 0.0 if outside peak window
    for h in morning_peak_window:
        if h not in cheapest_before_morning:  # Don't discharge in hours we're charging
            plan[h] = 1.0  # Discharge (action=1 in peak)

    # Phase 3: Charge in cheapest hours before evening peak
    # Action = 1.0 if outside peak window (charge), 0.0 if in peak window
    for h in cheapest_before_evening:
        # Always charge in cheapest hours (they should be outside peak windows)
        plan[h] = 1.0  # Charge (action=1 outside peak)

    # Phase 4: Discharge fully during evening peak window (peak-1 to peak+1)
    # Action = 1.0 if in peak window (discharge), 0.0 if outside peak window
    for h in evening_peak_window:
        if h not in cheapest_before_evening:  # Don't discharge in hours we're charging
            plan[h] = 1.0  # Discharge (action=1 in peak)

    # No Phase 5: Do not charge back to 0.5 - let battery end at whatever SOC after evening discharge

    return plan


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
    "pv_da_two_cycles_plan",
]



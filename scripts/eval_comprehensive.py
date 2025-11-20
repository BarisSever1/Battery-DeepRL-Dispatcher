"""Comprehensive evaluation script comparing TD3 agent vs baseline policies."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.policies import (
    pv_da_plan,
    pv_ds3_plan,
)
from src.envs import BESSEnv
from src.rl.td3 import TD3Agent, TD3Config


@dataclass
class HourlyData:
    """Comprehensive hourly metrics for a single episode."""

    # Basic identifiers
    hour: int
    
    # Actions and decisions
    action: float  # Agent action (delta)
    delta: float  # Actual delta used (may differ from action due to constraints)
    is_discharge_slot: bool  # Whether this hour allows discharge
    
    # Battery state
    soc: float
    soc_before: float  # SOC before this step
    dod_morning: float
    dod_evening: float
    
    # Power flows (MW)
    p_battery: float  # Battery power (positive=discharge, negative=charge)
    p_pv_raw: float  # Raw PV generation
    p_pv_grid: float  # PV power to grid
    p_bess_em: float  # BESS power to/from energy market (positive=export, negative=import)
    p_reserve: float  # Reserve capacity provided
    
    # Prices (EUR/MWh or EUR/MW·h)
    price_em: float  # Energy market price
    price_as: float  # Ancillary services (reserve) price
    
    # Revenue and costs (EUR)
    revenue_pv: float  # Revenue from PV to grid
    revenue_energy: float  # Revenue from energy market (negative if buying)
    revenue_reserve: float  # Revenue from reserve provision
    degradation_cost: float  # Battery degradation cost
    reward: float  # Step reward (revenue - degradation)
    
    # Cumulative metrics (EUR)
    cumulative_revenue: float
    cumulative_degradation: float
    cumulative_profit: float
    
    # Observation features (denormalized - raw values)
    k: float  # Hour of day (0-23)
    weekday: float  # Day of week (0-6)
    season: float  # Season (0-3)
    price_em_raw: float  # Energy price (EUR/MWh, raw)
    price_as_raw: float  # Reserve price (EUR/MW·h, raw)
    p_res_total_raw: float  # PV generation (MW, raw)
    soc_raw: float  # SOC (0-1, raw)
    dod_raw: float  # DOD (0-1, raw)
    price_em_max_morning: float  # Morning max price (EUR/MWh, raw)
    price_em_max_evening: float  # Evening max price (EUR/MWh, raw)
    k_em_max_morning: float  # Morning peak hour (0-23, raw)
    k_em_max_evening: float  # Evening peak hour (0-23, raw)
    price_em_min: float  # Daily min price (EUR/MWh, raw)
    k_em_min: float  # Daily min hour (0-23, raw)
    price_as_min: float  # Daily reserve min (EUR/MW·h, raw)
    price_as_max: float  # Daily reserve max (EUR/MW·h, raw)
    time_to_peak_hour: float


@dataclass
class ComprehensiveStats:
    """Comprehensive episode statistics for evaluation."""

    # Financial
    profit: float
    revenue: float
    revenue_pv: float
    revenue_energy: float
    revenue_reserve: float
    degradation_cost: float

    # Battery State
    soc_min: float
    soc_max: float
    soc_mean: float
    soc_final: float
    dod_max_morning: float
    dod_max_evening: float
    dod_max_overall: float

    # Throughput
    energy_throughput_mwh: float  # Total energy cycled
    charge_cycles: float  # Approximate number of cycles

    # Operational
    action_mean: float
    discharge_hours: int  # Hours with discharge (p_battery < 0)


def load_test_dates(test_data_path: str) -> List:
    """Load unique dates from test data parquet file (returns date objects, not strings)."""
    df = pd.read_parquet(test_data_path)
    if "date" in df.columns:
        # If date column exists, convert to date objects
        if df["date"].dtype == "object":
            dates = sorted([pd.to_datetime(d).date() for d in df["date"].unique()])
        else:
            dates = sorted(df["date"].unique())
    elif "datetime" in df.columns:
        dates = sorted(df["datetime"].dt.date.unique())
    else:
        raise ValueError("Test data must have 'date' or 'datetime' column")
    return dates


def denormalize_observation_features(
    env: BESSEnv, obs_vec: np.ndarray, soc: float, dod: Optional[float] = None
) -> Dict[str, float]:
    """
    Denormalize the core observation features (first 17) to raw values.
    
    NOTE: This function is ONLY for CSV output/display purposes. The agent and
    environment continue to use normalized values for all training and evaluation.
    This conversion happens AFTER the step, so it doesn't affect any results.
    
    Args:
        env: BESSEnv instance with denormalization methods
        obs_vec: Normalized observation vector (>=16 dims). If the environment
                 appends extra features (e.g., a discharge-slot indicator), they
                 are ignored here for denormalization.
        soc: Current SOC (already raw, 0-1) - used to verify/override denormalized SOC
        dod: Optional current DOD (if None, will denormalize from observation)
    
    Returns:
        Dictionary of denormalized feature values (for display only)
    """
    # Feature names in order of observation vector
    feature_names = [
        "k",
        "weekday",
        "season",
        "price_em",
        "price_as",
        "p_res_total",
        "soc",  # Special: normalized separately
        "dod",  # Special: normalized separately
        "price_em_max_morning",
        "price_em_max_evening",
        "k_em_max_morning",
        "k_em_max_evening",
        "price_em_min",
        "k_em_min",
        "price_as_min",
        "price_as_max",
        "time_to_peak_hour",
    ]
    
    denormalized = {}
    
    for i, feature_name in enumerate(feature_names):
        norm_value = float(obs_vec[i])
        
        if feature_name == "soc":
            # Inverse of: normalized = 2.0 * soc - 1.0
            # Use provided soc if available (more accurate), otherwise denormalize
            denormalized["soc_raw"] = soc if soc is not None else (norm_value + 1.0) / 2.0
        elif feature_name == "dod":
            # Inverse of: normalized = 2.0 * dod - 1.0
            # Use provided dod if available, otherwise denormalize from observation
            if dod is not None:
                denormalized["dod_raw"] = dod
            else:
                denormalized["dod_raw"] = (norm_value + 1.0) / 2.0
        else:
            # Denormalize using environment's method
            raw_value = env._denormalize(feature_name, norm_value)
            
            # Map to output field names
            if feature_name == "price_em":
                denormalized["price_em_raw"] = raw_value
            elif feature_name == "price_as":
                denormalized["price_as_raw"] = raw_value
            elif feature_name == "p_res_total":
                denormalized["p_res_total_raw"] = raw_value
            else:
                denormalized[feature_name] = raw_value
    
    return denormalized


def compute_action_mask(env: BESSEnv) -> Optional[Tuple[float, float]]:
    """Infer action bounds from the environment's current peak-window configuration."""
    hour = getattr(env, "current_hour", None)
    if hour is None:
        return None

    def _to_set(hours):
        if hours is None:
            return set()
        if isinstance(hours, set):
            return hours
        return set(hours)

    morning_hours = _to_set(getattr(env, "morning_peak_hours", None))
    evening_hours = _to_set(getattr(env, "evening_peak_hours", None))

    if not morning_hours and not evening_hours:
        return None

    in_peak = (hour in morning_hours) or (hour in evening_hours)
    return (0.0, 1.0) if in_peak else (-1.0, 0.0)


def evaluate_td3_agent(
    env: BESSEnv,
    agent: TD3Agent,
    test_dates: List[str],
    seed: int = 999,
) -> Tuple[List[ComprehensiveStats], List[List[HourlyData]]]:
    """Evaluate TD3 agent on test dates and return comprehensive statistics and hourly data."""
    stats_list: List[ComprehensiveStats] = []
    hourly_data_list: List[List[HourlyData]] = []
    np.random.seed(seed)

    for date in test_dates:
        # Reset environment with specific date
        obs, _ = env.reset(seed=seed, options={"date": date})
        obs_vec = np.asarray(obs, dtype=np.float32)

        # Track metrics throughout episode
        soc_values: List[float] = []
        dod_morning_values: List[float] = []
        dod_evening_values: List[float] = []
        action_values: List[float] = []
        p_battery_values: List[float] = []
        hourly_data: List[HourlyData] = []

        totals = {
            "revenue_pv": 0.0,
            "revenue_energy": 0.0,
            "revenue_reserve": 0.0,
            "degradation": 0.0,
        }

        done = False
        step = 0
        final_info: Dict[str, float] = {}

        # Maintain history of observations so the LSTM sees a sequence [1, t, F]
        obs_history: List[np.ndarray] = []

        while not done:
            # Append current observation to history and feed full sequence to the actor
            obs_history.append(obs_vec)
            # Truncate to 12 for consistency with training sequence length
            if len(obs_history) > 12:
                obs_history.pop(0)
            state_seq = np.asarray(obs_history, dtype=np.float32)[np.newaxis, :, :]  # [1, T, F]
            mask = compute_action_mask(env)
            action_seq = agent.act(state_seq, eval_mode=True, action_mask=mask)  # [1, T, 1]
            action = float(action_seq[0, -1, 0])  # take action for the latest timestep
            action_values.append(float(action))

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(
                np.array([action], dtype=np.float32)
            )

            # Collect metrics
            hour = info.get("hour", step)
            soc = float(info.get("soc", env.soc))
            soc_before = float(info.get("soc_before", env.soc))
            soc_values.append(soc)
            dod_morning = float(info.get("dod_morning", 0.0))
            dod_evening = float(info.get("dod_evening", 0.0))
            dod_morning_values.append(dod_morning)
            dod_evening_values.append(dod_evening)
            p_battery = float(info.get("p_battery", 0.0))
            p_battery_values.append(p_battery)
            
            # IMPORTANT: Denormalize observation features ONLY for CSV output/display
            # The agent and environment still use normalized values internally.
            # This denormalization happens AFTER the step, so it doesn't affect evaluation.
            # obs_vec contains the normalized state the agent saw before taking the action
            obs_features = denormalize_observation_features(env, obs_vec, soc_before, None)

            # Update cumulative totals
            totals["revenue_pv"] += info.get("revenue_pv_grid", 0.0)
            totals["revenue_energy"] += info.get("revenue_energy", 0.0)
            totals["revenue_reserve"] += info.get("revenue_reserve", 0.0)
            totals["degradation"] += info.get("cost_degradation", 0.0)
            cumulative_revenue = (
                totals["revenue_pv"] + totals["revenue_energy"] + totals["revenue_reserve"]
            )
            cumulative_degradation = totals["degradation"]
            cumulative_profit = cumulative_revenue - cumulative_degradation

            # Store comprehensive hourly data
            hourly_data.append(
                HourlyData(
                    hour=hour,
                    action=float(action),
                    delta=float(info.get("delta", action)),
                    is_discharge_slot=bool(info.get("is_discharge_slot", False)),
                    soc=soc,
                    soc_before=soc_before,
                    dod_morning=dod_morning,
                    dod_evening=dod_evening,
                    p_battery=p_battery,
                    p_pv_raw=float(info.get("p_pv_raw", 0.0)),
                    p_pv_grid=float(info.get("p_pv_grid", 0.0)),
                    p_bess_em=float(info.get("p_bess_em", 0.0)),
                    p_reserve=float(info.get("p_reserve", 0.0)),
                    price_em=float(info.get("price_em", 0.0)),
                    price_as=float(info.get("price_as", 0.0)),
                    revenue_pv=float(info.get("revenue_pv_grid", 0.0)),
                    revenue_energy=float(info.get("revenue_energy", 0.0)),
                    revenue_reserve=float(info.get("revenue_reserve", 0.0)),
                    degradation_cost=float(info.get("cost_degradation", 0.0)),
                    reward=float(reward),
                    cumulative_revenue=cumulative_revenue,
                    cumulative_degradation=cumulative_degradation,
                    cumulative_profit=cumulative_profit,
                    **obs_features,
                )
            )

            final_info = info
            obs_vec = np.asarray(next_obs, dtype=np.float32)
            done = terminated or truncated
            step += 1

        # Calculate comprehensive statistics
        revenue = (
            totals["revenue_pv"] + totals["revenue_energy"] + totals["revenue_reserve"]
        )
        profit = revenue - totals["degradation"]

        # SOC statistics
        soc_min = float(np.min(soc_values)) if soc_values else 0.5
        soc_max = float(np.max(soc_values)) if soc_values else 0.5
        soc_mean = float(np.mean(soc_values)) if soc_values else 0.5
        soc_final = float(soc_values[-1]) if soc_values else 0.5

        # DOD statistics
        dod_max_morning = float(np.max(dod_morning_values)) if dod_morning_values else 0.0
        dod_max_evening = float(np.max(dod_evening_values)) if dod_evening_values else 0.0
        dod_max_overall = max(dod_max_morning, dod_max_evening)

        # Energy throughput (MWh)
        dt = env.dt  # hours
        energy_throughput_mwh = float(
            np.sum(np.abs(p_battery_values)) * dt
        )  # Sum of |power| * dt

        # Estimate cycles: throughput / (capacity * avg_dod)
        # If no DOD, assume shallow cycle (DOD = 0.1 for safety)
        avg_dod = max(dod_max_overall, 0.1) if dod_max_overall > 0 else 0.1
        capacity_mwh = float(env.E_capacity)
        charge_cycles = energy_throughput_mwh / (capacity_mwh * avg_dod) if avg_dod > 0 else 0.0

        # Operational metrics
        action_mean = float(np.mean(action_values)) if action_values else 0.0
        discharge_hours = sum(1 for p in p_battery_values if p < 0)

        stats = ComprehensiveStats(
            profit=profit,
            revenue=revenue,
            revenue_pv=totals["revenue_pv"],
            revenue_energy=totals["revenue_energy"],
            revenue_reserve=totals["revenue_reserve"],
            degradation_cost=totals["degradation"],
            soc_min=soc_min,
            soc_max=soc_max,
            soc_mean=soc_mean,
            soc_final=soc_final,
            dod_max_morning=dod_max_morning,
            dod_max_evening=dod_max_evening,
            dod_max_overall=dod_max_overall,
            energy_throughput_mwh=energy_throughput_mwh,
            charge_cycles=charge_cycles,
            action_mean=action_mean,
            discharge_hours=discharge_hours,
        )

        stats_list.append(stats)
        hourly_data_list.append(hourly_data)

    return stats_list, hourly_data_list


def evaluate_baseline_comprehensive(
    env: BESSEnv,
    plan_fn: Callable[[BESSEnv, int], List[float]],
    test_dates: List[str],
    seed: int = 999,
) -> Tuple[List[ComprehensiveStats], List[List[HourlyData]]]:
    """Evaluate baseline policy on test dates and return comprehensive statistics and hourly data."""
    stats_list: List[ComprehensiveStats] = []
    hourly_data_list: List[List[HourlyData]] = []
    np.random.seed(seed)

    for date in test_dates:
        # Reset environment with specific date
        obs, _ = env.reset(seed=seed, options={"date": date})
        obs_vec = np.asarray(obs, dtype=np.float32)

        # Get plan from baseline policy
        plan = plan_fn(env, seed)

        # Track metrics throughout episode
        soc_values: List[float] = []
        dod_morning_values: List[float] = []
        dod_evening_values: List[float] = []
        action_values: List[float] = []
        p_battery_values: List[float] = []
        hourly_data: List[HourlyData] = []

        totals = {
            "revenue_pv": 0.0,
            "revenue_energy": 0.0,
            "revenue_reserve": 0.0,
            "degradation": 0.0,
        }

        final_info: Dict[str, float] = {}

        for hour, delta in enumerate(plan):
            action = np.array([float(np.clip(delta, -1.0, 1.0))], dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Collect metrics
            soc = float(info.get("soc", env.soc))
            soc_before = float(info.get("soc_before", env.soc))
            soc_values.append(soc)
            dod_morning = float(info.get("dod_morning", 0.0))
            dod_evening = float(info.get("dod_evening", 0.0))
            dod_morning_values.append(dod_morning)
            dod_evening_values.append(dod_evening)
            p_battery = float(info.get("p_battery", 0.0))
            p_battery_values.append(p_battery)
            action_values.append(float(delta))
            
            # IMPORTANT: Denormalize observation features ONLY for CSV output/display
            # The agent and environment still use normalized values internally.
            # This denormalization happens AFTER the step, so it doesn't affect evaluation.
            # obs_vec contains the normalized state the agent saw before taking the action
            obs_features = denormalize_observation_features(env, obs_vec, soc_before, None)

            # Update cumulative totals
            totals["revenue_pv"] += info.get("revenue_pv_grid", 0.0)
            totals["revenue_energy"] += info.get("revenue_energy", 0.0)
            totals["revenue_reserve"] += info.get("revenue_reserve", 0.0)
            totals["degradation"] += info.get("cost_degradation", 0.0)
            cumulative_revenue = (
                totals["revenue_pv"] + totals["revenue_energy"] + totals["revenue_reserve"]
            )
            cumulative_degradation = totals["degradation"]
            cumulative_profit = cumulative_revenue - cumulative_degradation

            # Store comprehensive hourly data
            hourly_data.append(
                HourlyData(
                    hour=hour,
                    action=float(delta),
                    delta=float(info.get("delta", delta)),
                    is_discharge_slot=bool(info.get("is_discharge_slot", False)),
                    soc=soc,
                    soc_before=soc_before,
                    dod_morning=dod_morning,
                    dod_evening=dod_evening,
                    p_battery=p_battery,
                    p_pv_raw=float(info.get("p_pv_raw", 0.0)),
                    p_pv_grid=float(info.get("p_pv_grid", 0.0)),
                    p_bess_em=float(info.get("p_bess_em", 0.0)),
                    p_reserve=float(info.get("p_reserve", 0.0)),
                    price_em=float(info.get("price_em", 0.0)),
                    price_as=float(info.get("price_as", 0.0)),
                    revenue_pv=float(info.get("revenue_pv_grid", 0.0)),
                    revenue_energy=float(info.get("revenue_energy", 0.0)),
                    revenue_reserve=float(info.get("revenue_reserve", 0.0)),
                    degradation_cost=float(info.get("cost_degradation", 0.0)),
                    reward=float(reward),
                    cumulative_revenue=cumulative_revenue,
                    cumulative_degradation=cumulative_degradation,
                    cumulative_profit=cumulative_profit,
                    **obs_features,
                )
            )

            final_info = info
            obs_vec = np.asarray(next_obs, dtype=np.float32)
            if terminated or truncated:
                break

        # Calculate comprehensive statistics (same as TD3 evaluation)
        revenue = (
            totals["revenue_pv"] + totals["revenue_energy"] + totals["revenue_reserve"]
        )
        profit = revenue - totals["degradation"]

        soc_min = float(np.min(soc_values)) if soc_values else 0.5
        soc_max = float(np.max(soc_values)) if soc_values else 0.5
        soc_mean = float(np.mean(soc_values)) if soc_values else 0.5
        soc_final = float(soc_values[-1]) if soc_values else 0.5

        dod_max_morning = float(np.max(dod_morning_values)) if dod_morning_values else 0.0
        dod_max_evening = float(np.max(dod_evening_values)) if dod_evening_values else 0.0
        dod_max_overall = max(dod_max_morning, dod_max_evening)

        dt = env.dt
        energy_throughput_mwh = float(np.sum(np.abs(p_battery_values)) * dt)

        avg_dod = max(dod_max_overall, 0.1) if dod_max_overall > 0 else 0.1
        capacity_mwh = float(env.E_capacity)
        charge_cycles = energy_throughput_mwh / (capacity_mwh * avg_dod) if avg_dod > 0 else 0.0

        action_mean = float(np.mean(action_values)) if action_values else 0.0
        discharge_hours = sum(1 for p in p_battery_values if p < 0)

        stats = ComprehensiveStats(
            profit=profit,
            revenue=revenue,
            revenue_pv=totals["revenue_pv"],
            revenue_energy=totals["revenue_energy"],
            revenue_reserve=totals["revenue_reserve"],
            degradation_cost=totals["degradation"],
            soc_min=soc_min,
            soc_max=soc_max,
            soc_mean=soc_mean,
            soc_final=soc_final,
            dod_max_morning=dod_max_morning,
            dod_max_evening=dod_max_evening,
            dod_max_overall=dod_max_overall,
            energy_throughput_mwh=energy_throughput_mwh,
            charge_cycles=charge_cycles,
            action_mean=action_mean,
            discharge_hours=discharge_hours,
        )

        stats_list.append(stats)
        hourly_data_list.append(hourly_data)

    return stats_list, hourly_data_list


def print_summary_table(
    results: Dict[str, List[ComprehensiveStats]],
    test_dates: List[str],
) -> None:
    """Print formatted summary table comparing all policies."""
    print("=" * 100)
    print("COMPREHENSIVE EVALUATION: TD3 vs Baselines")
    print("=" * 100)
    print(f"Test Data: {len(test_dates)} days")
    print()

    # Calculate means for each policy
    summary_data = []
    for policy_name, stats_list in results.items():
        if not stats_list:
            continue

        mean_stats = {
            "policy": policy_name,
            "profit": np.mean([s.profit for s in stats_list]),
            "revenue": np.mean([s.revenue for s in stats_list]),
            "degradation": np.mean([s.degradation_cost for s in stats_list]),
            "soc_min": np.mean([s.soc_min for s in stats_list]),
            "soc_max": np.mean([s.soc_max for s in stats_list]),
            "soc_final": np.mean([s.soc_final for s in stats_list]),
            "dod_max": np.mean([s.dod_max_overall for s in stats_list]),
            "throughput": np.mean([s.energy_throughput_mwh for s in stats_list]),
            "cycles": np.mean([s.charge_cycles for s in stats_list]),
            "deg_per_mwh": np.mean(
                [
                    s.degradation_cost / max(s.energy_throughput_mwh, 0.01)
                    for s in stats_list
                ]
            ),
            "avg_dod": np.mean([s.dod_max_overall for s in stats_list]),
        }
        summary_data.append(mean_stats)

    # Print main table
    print(
        f"{'Policy':<25} | {'Profit':>10} | {'Revenue':>10} | {'Degradation':>12} | "
        f"{'SOC Range':>12} | {'Max DOD':>8} | {'Throughput':>12} | {'Cycles/Day':>11}"
    )
    print("-" * 100)

    for data in summary_data:
        soc_range = f"{data['soc_min']:.2f}-{data['soc_max']:.2f}"
        print(
            f"{data['policy']:<25} | {data['profit']:>10.2f} | {data['revenue']:>10.2f} | "
            f"{data['degradation']:>12.2f} | {soc_range:>12} | {data['dod_max']:>8.3f} | "
            f"{data['throughput']:>11.2f} MWh | {data['cycles']:>10.2f}"
        )

    print()
    print("Battery Wear Analysis:")
    for data in summary_data:
        print(
            f"  {data['policy']:<25}: {data['deg_per_mwh']:>6.2f} EUR/MWh throughput, "
            f"avg DOD {data['avg_dod']:.3f}"
        )

    print("=" * 100)


def save_results(
    results: Dict[str, List[ComprehensiveStats]],
    hourly_results: Dict[str, List[List[HourlyData]]],
    test_dates: List[str],
    output_dir: Path,
) -> None:
    """Save evaluation results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed per-episode CSV
    rows = []
    for policy_name, stats_list in results.items():
        for i, stats in enumerate(stats_list):
            row = {"policy": policy_name, "test_day": test_dates[i] if i < len(test_dates) else f"day_{i}"}
            row.update(asdict(stats))
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "comprehensive_evaluation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved detailed results to: {csv_path}")

    # Save hourly data CSV (for plotting)
    hourly_rows = []
    for policy_name, hourly_data_list in hourly_results.items():
        for day_idx, hourly_data in enumerate(hourly_data_list):
            test_day = test_dates[day_idx] if day_idx < len(test_dates) else f"day_{day_idx}"
            for hd in hourly_data:
                row = {
                    "policy": policy_name,
                    "test_day": test_day,
                }
                row.update(asdict(hd))
                hourly_rows.append(row)

    hourly_df = pd.DataFrame(hourly_rows)
    hourly_csv_path = output_dir / "hourly_data.csv"
    hourly_df.to_csv(hourly_csv_path, index=False)
    print(f"[OK] Saved hourly data to: {hourly_csv_path} ({len(hourly_df)} rows)")

    # Save summary JSON
    summary = {}
    for policy_name, stats_list in results.items():
        if not stats_list:
            continue
        summary[policy_name] = {
            "mean_profit": float(np.mean([s.profit for s in stats_list])),
            "mean_revenue": float(np.mean([s.revenue for s in stats_list])),
            "mean_degradation": float(np.mean([s.degradation_cost for s in stats_list])),
            "mean_soc_min": float(np.mean([s.soc_min for s in stats_list])),
            "mean_soc_max": float(np.mean([s.soc_max for s in stats_list])),
            "mean_soc_final": float(np.mean([s.soc_final for s in stats_list])),
            "mean_dod_max": float(np.mean([s.dod_max_overall for s in stats_list])),
            "mean_throughput_mwh": float(np.mean([s.energy_throughput_mwh for s in stats_list])),
            "mean_cycles": float(np.mean([s.charge_cycles for s in stats_list])),
        }

    json_path = output_dir / "evaluation_summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved summary to: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation: TD3 agent vs baseline policies"
    )
    parser.add_argument(
        "--actor-checkpoint",
        type=Path,
        default=Path("checkpoints/td3_nonlinear"),
        help="Path to TD3 actor checkpoint file OR a directory. If a directory is provided, "
             "the latest actor_ep*.pth will be selected automatically.",
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=Path("data/processed/training_features_normalized_test.parquet"),
        help="Path to test data parquet file",
    )
    parser.add_argument(
        "--degradation-model",
        type=str,
        default="nonlinear",
        choices=["nonlinear", "linear"],
        help="Degradation cost model used by the environment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=999,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-episode results",
    )

    args = parser.parse_args()

    # Helper: resolve latest actor checkpoint if a directory (or non-existent file) is passed
    def _resolve_latest_actor(path: Path) -> Path:
        if path.is_file():
            return path
        search_dir = path if path.is_dir() else path.parent
        candidates = list(search_dir.glob("actor_ep*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No actor checkpoints matching 'actor_ep*.pth' found in {search_dir}")
        import re
        def ep_num(p: Path) -> int:
            m = re.search(r"actor_ep(\d+)\.pth$", p.name)
            return int(m.group(1)) if m else -1
        return max(candidates, key=ep_num)

    # Load test dates
    print(f"Loading test data from: {args.test_data_path}")
    test_dates = load_test_dates(str(args.test_data_path))
    print(f"Found {len(test_dates)} test days: {test_dates}")

    # Initialize environment
    env = BESSEnv(
        data_path=str(args.test_data_path),
        degradation_model=args.degradation_model,
    )
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    # Initialize and load TD3 agent
    resolved_actor = _resolve_latest_actor(args.actor_checkpoint)
    print(f"\nLoading TD3 agent from: {resolved_actor}")
    config = TD3Config(
        action_low=float(env.action_space.low.min()),
        action_high=float(env.action_space.high.max()),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        device=device,
    )
    agent.actor.load_state_dict(torch.load(resolved_actor, map_location=device))
    agent.actor.eval()
    print("[OK] Agent loaded successfully")

    # Evaluate all policies
    results: Dict[str, List[ComprehensiveStats]] = {}
    hourly_results: Dict[str, List[List[HourlyData]]] = {}

    print("\n" + "=" * 100)
    print("Evaluating policies...")
    print("=" * 100)

    # TD3 Agent
    print("\nEvaluating TD3 (Nonlinear)...")
    stats_list, hourly_list = evaluate_td3_agent(env, agent, test_dates, args.seed)
    results["TD3 (Nonlinear)"] = stats_list
    hourly_results["TD3 (Nonlinear)"] = hourly_list
    print(f"[OK] Completed {len(stats_list)} episodes")

    # Baselines
    baselines = [
        ("PV+DS3 (Reserve Only)", pv_ds3_plan),
        ("PV+DA (Heuristic)", pv_da_plan),
    ]

    for name, plan_fn in baselines:
        print(f"\nEvaluating {name}...")
        stats_list, hourly_list = evaluate_baseline_comprehensive(env, plan_fn, test_dates, args.seed)
        results[name] = stats_list
        hourly_results[name] = hourly_list
        print(f"[OK] Completed {len(stats_list)} episodes")

    # Print summary
    print()
    print_summary_table(results, test_dates)

    # Save results
    save_results(results, hourly_results, test_dates, args.output_dir)

    print("\n[OK] Evaluation complete!")


if __name__ == "__main__":
    main()


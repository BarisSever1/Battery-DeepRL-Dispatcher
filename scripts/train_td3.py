"""Training entry point for TD3 + LSTM agent on the BESS environment."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
import torch
from gymnasium.utils import seeding

from src.envs import BESSEnv
from src.rl.buffer import ReplayBuffer
from src.rl.td3 import TD3Agent, TD3Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TD3+LSTM agent on BESSEnv")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--days-per-epoch", type=int, default=20, help="Episodes per epoch")
    parser.add_argument("--warmup-steps", type=int, default=5000, help="Warmup steps with random actions")
    parser.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size for updates")
    parser.add_argument("--updates-per-step", type=int, default=1, help="Gradient updates per environment step")
    parser.add_argument("--seq-len", type=int, default=24, help="Sequence length sampled from replay buffer")
    parser.add_argument("--exploration-std", type=float, default=0.5, help="Std dev of exploration noise")
    parser.add_argument("--target-std", type=float, default=0.5, help="Std dev of target policy smoothing noise")
    parser.add_argument("--noise-clip", type=float, default=0.4, help="Clamp for target policy smoothing noise")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Polyak averaging factor (faster target updates to track policy changes)")
    parser.add_argument("--policy-delay", type=int, default=2, help="Delayed policy update interval")
    parser.add_argument("--lr", type=float, default=7e-5, help="Learning rate for actor/critic")
    parser.add_argument("--actor-lr", type=float, default=None, help="Optional override for actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=None, help="Optional override for critic learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Training device: auto|cpu|cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Print metrics every N episodes")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory for model checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Episodes between checkpoints")
    parser.add_argument("--csv-log", type=Path, default=None, help="Optional path to append episode metrics CSV")
    parser.add_argument("--eval-mode", action="store_true", help="Disable exploration noise during training (debug)")
    parser.add_argument(
        "--degradation-model",
        type=str,
        default="nonlinear",
        choices=["nonlinear", "linear"],
        help="Degradation cost model used by the environment.",
    )
    parser.add_argument(
        "--quick-eval-interval",
        type=int,
        default=0,
        help="Run a 1-day rollout every N episodes and save hourly CSV (0=off)",
    )
    parser.add_argument(
        "--quick-eval-date",
        type=str,
        default=None,
        help="YYYY-MM-DD date to rollout (required if --quick-eval-interval>0)",
    )
    parser.add_argument(
        "--quick-eval-outdir",
        type=Path,
        default=Path("results/quick"),
        help="Directory to save quick hourly rollout CSVs",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/training_features_normalized_train.parquet",
        help="Path to training data parquet file",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CSVWriter:
    """Wrapper for csv.writer that holds the file reference."""
    def __init__(self, writer: csv.writer, file):
        self.writer = writer
        self.file = file
    
    def writerow(self, row):
        return self.writer.writerow(row)


def initialize_csv(path: Path) -> Optional[CSVWriter]:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    csv_file = path.open("a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(
            [
                "episode",
                "return",
                "episode_profit",
                "episode_revenue",
                "episode_degradation",
                "rev_pv",
                "rev_energy",
                "rev_reserve",
                "deg_cost",
            ]
        )
    return CSVWriter(writer, csv_file)


def close_csv(writer: Optional[CSVWriter]) -> None:
    if writer is not None:
        writer.file.close()


def _denorm_obs_features(env: BESSEnv, obs_vec: np.ndarray, soc_before: float) -> Dict[str, float]:
    """Denormalize core features for CSV output (mirrors eval script semantics)."""
    feature_names = [
        "k", "weekday", "season", "price_em", "price_as", "p_res_total",
        "soc", "dod", "price_em_max_morning", "price_em_max_evening",
        "k_em_max_morning", "k_em_max_evening", "price_em_min", "k_em_min",
        "price_as_min", "price_as_max",
        "time_to_peak_hour",
    ]
    out: Dict[str, float] = {}
    for i, name in enumerate(feature_names):
        val = float(obs_vec[i])
        if name == "soc":
            out["soc_raw"] = soc_before
        elif name == "dod":
            # We don't track live DOD here; restore from normalized as a proxy
            out["dod_raw"] = (val + 1.0) / 2.0
        else:
            raw = env._denormalize(name, val)
            if name == "price_em":
                out["price_em_raw"] = raw
            elif name == "price_as":
                out["price_as_raw"] = raw
            elif name == "p_res_total":
                out["p_res_total_raw"] = raw
            else:
                out[name] = raw
    return out


def guided_warmup_action(env: BESSEnv, obs_vec: np.ndarray, random_prob: float = 0.3) -> np.ndarray:
    """
    During warmup: mix rule-based hints (70%) with random actions (30%).
    This ensures the agent sees both good behaviors and full action space coverage.
    
    Args:
        env: The BESS environment
        obs_vec: Normalized observation vector (18-dim)
        random_prob: Probability of using pure random action (default 0.3)
    
    Returns:
        Action array of shape (1,)
    """
    if random.random() < random_prob:
        # 30% pure random to ensure full action space coverage
        return env.action_space.sample().astype(np.float32)
    
    # 70% rule-based hints
    try:
        # Extract normalized features from observation
        hour_norm = obs_vec[0]  # k (hour of day)
        soc_norm = obs_vec[6]   # SOC
        
        # Denormalize to get actual values
        hour = int(round(env._denormalize("k", hour_norm)))
        hour = max(0, min(23, hour))  # Clamp to valid range
        soc = env._denormalize("soc", soc_norm)
        soc = max(0.0, min(1.0, soc))  # Clamp to valid range
        
        # Get peak/cheap hours from environment
        cheapest_hours = getattr(env, "cheapest_hours", set())
        peak_hours = getattr(env, "peak_hours", set())
        
        # Rule-based hints
        if hour in cheapest_hours and soc < 0.7:
            # Hint: charge during cheap hours if SOC is low
            # Use moderate to strong charging
            action = np.random.uniform(-1.0, -0.5, size=(1,)).astype(np.float32)
        elif hour in peak_hours and soc > 0.3:
            # Hint: discharge during peak hours if SOC is high
            # Use moderate to strong discharging
            action = np.random.uniform(0.5, 1.0, size=(1,)).astype(np.float32)
        else:
            # Mid-price or constraints: use random to explore
            action = env.action_space.sample().astype(np.float32)
        
        return action
    except Exception:
        # Fallback to random if anything goes wrong
        return env.action_space.sample().astype(np.float32)


def quick_rollout(env: BESSEnv, agent: TD3Agent, rollout_date: str) -> List[Dict[str, float]]:
    """Run a single-day deterministic rollout, return hourly rows (like eval)."""
    import numpy as _np
    rows: List[Dict[str, float]] = []
    obs, _ = env.reset(options={"date": _np.datetime64(rollout_date).astype("datetime64[D]").astype(object)})
    obs_vec = _np.asarray(obs, dtype=_np.float32)
    obs_history: List[_np.ndarray] = []
    done = False
    step = 0
    # Reset hidden state at episode start
    agent.reset_hidden_state()
    totals = {"revenue_pv": 0.0, "revenue_energy": 0.0, "revenue_reserve": 0.0, "degradation": 0.0}

    history_len = 24
    while not done:
        # RL clean-up patch: keep fixed-length history (24)
        obs_history.append(obs_vec)
        if len(obs_history) > history_len:
            obs_history.pop(0)
        state_seq = _np.asarray(obs_history, dtype=_np.float32)[ _np.newaxis, :, :]  # [1,T,F]
        # Hidden state is already reset at episode start, no need to reset again
        action_vec = agent.act(state_seq, eval_mode=True, reset_hidden=False)[0]
        action_arr = _np.asarray(action_vec, dtype=_np.float32)

        q1_value = float("nan")
        q2_value = float("nan")
        try:
            with torch.no_grad():
                state_seq_t = torch.as_tensor(state_seq, dtype=torch.float32, device=agent.device)
                action_last_t = torch.as_tensor(action_arr.reshape(1, 1, -1), dtype=torch.float32, device=agent.device)
                action_seq_t = action_last_t.expand(-1, state_seq_t.shape[1], -1)
                q1_eval, q2_eval = agent.critic(state_seq_t, action_seq_t)
                q1_value = float(q1_eval.outputs[:, -1, :].view(1, -1).mean().item())
                q2_value = float(q2_eval.outputs[:, -1, :].view(1, -1).mean().item())
        except Exception:
            q1_value = float("nan")
            q2_value = float("nan")
        next_obs, reward, terminated, truncated, info = env.step(action_arr)

        soc = float(info.get("soc", env.soc))
        soc_before = float(info.get("soc_before", env.soc))

        # Cumulative trackers
        totals["revenue_pv"] += info.get("revenue_pv_grid", 0.0)
        totals["revenue_energy"] += info.get("revenue_energy", 0.0)
        totals["revenue_reserve"] += info.get("revenue_reserve", 0.0)
        totals["degradation"] += info.get("cost_degradation", 0.0)
        cumulative_revenue = totals["revenue_pv"] + totals["revenue_energy"] + totals["revenue_reserve"]
        cumulative_degradation = totals["degradation"]
        cumulative_profit = cumulative_revenue - cumulative_degradation

        # Denormalized feature snapshot
        obs_features = _denorm_obs_features(env, obs_vec, soc_before)

        action_scalar = float(action_arr[0]) if action_arr.size > 0 else float(action_arr)
        rows.append({
            "hour": info.get("hour", step),
            "action": action_scalar,
            "delta": float(info.get("delta", action_scalar)),
            "soc": soc,
            "soc_before": soc_before,
            "dod_morning": float(info.get("dod_morning", 0.0)),
            "dod_evening": float(info.get("dod_evening", 0.0)),
            "p_battery": float(info.get("p_battery", 0.0)),
            "p_pv_raw": float(info.get("p_pv_raw", 0.0)),
            "p_pv_grid": float(info.get("p_pv_grid", 0.0)),
            "p_bess_em": float(info.get("p_bess_em", 0.0)),
            "p_reserve": float(info.get("p_reserve", 0.0)),
            "price_em": float(info.get("price_em", 0.0)),
            "price_as": float(info.get("price_as", 0.0)),
            "revenue_pv": float(info.get("revenue_pv_grid", 0.0)),
            "revenue_energy": float(info.get("revenue_energy", 0.0)),
            "revenue_reserve": float(info.get("revenue_reserve", 0.0)),
            "degradation_cost": float(info.get("cost_degradation", 0.0)),
            "reward": float(reward),
            "reward_base": float(info.get("reward_base", info.get("raw_reward", 0.0))),
            "reward_with_shaping": float(info.get("reward_with_shaping", info.get("raw_reward", 0.0) + info.get("reward_shaping", 0.0))),
            "reward_final": float(info.get("reward_final", reward)),
            "reward_shaping_component": float(info.get("reward_shaping", 0.0)),
            "q1_eval": q1_value,
            "q2_eval": q2_value,
            "cumulative_revenue": cumulative_revenue,
            "cumulative_degradation": cumulative_degradation,
            "cumulative_profit": cumulative_profit,
            "time_to_peak_hour_raw": float(info.get("time_to_peak_hour_raw", info.get("time_to_peak_hour", 0.0))),
            **obs_features,
        })

        obs_vec = _np.asarray(next_obs, dtype=_np.float32)
        done = terminated or truncated
        step += 1

    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    # Load optional training YAML for overrides (actor/critic LR, gamma, lr_schedule, etc.)
    training_yaml_path = Path("config/training.yaml")
    yaml_cfg = {}
    if training_yaml_path.exists():
        try:
            with training_yaml_path.open("r", encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: failed to read {training_yaml_path}: {e}")
    algo_cfg = (yaml_cfg.get("algorithm") or {}) if isinstance(yaml_cfg, dict) else {}
    train_cfg = (yaml_cfg.get("training") or {}) if isinstance(yaml_cfg, dict) else {}
    exploration_cfg = (yaml_cfg.get("exploration") or {}) if isinstance(yaml_cfg, dict) else {}
    
    # Gaussian exploration decay parameters
    initial_noise_std = float(exploration_cfg.get("initial_noise_std", 1.0)) if exploration_cfg else 1.0
    final_noise_std = float(exploration_cfg.get("final_noise_std", 0.1)) if exploration_cfg else 0.1
    noise_decay_episodes = int(exploration_cfg.get("noise_decay_episodes", 2000)) if exploration_cfg else 2000

    env = BESSEnv(data_path=args.data_path, degradation_model=args.degradation_model)
    env.action_space.seed(args.seed)

    env.np_random, _ = seeding.np_random(args.seed)

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(np.prod(env.action_space.shape))

    # Align days-per-epoch with the number of unique training days when unspecified or <= 0
    num_train_days = int(len(getattr(env, "dates", []))) if hasattr(env, "dates") else 0
    if num_train_days <= 0:
        raise RuntimeError("Training dataset has no days. Check --data-path and feature files.")
    # Always set days-per-epoch to the number of training days
    print(f"Setting days-per-epoch to number of training days: {num_train_days}")
    args.days_per_epoch = num_train_days

    # Allow YAML to override core algorithm knobs when present
    gamma_val = float(algo_cfg.get("gamma", args.gamma)) if algo_cfg else args.gamma
    tau_val = float(algo_cfg.get("tau", args.tau)) if algo_cfg else args.tau
    actor_lr_val = float(algo_cfg.get("actor_lr", args.actor_lr if args.actor_lr is not None else args.lr))
    critic_lr_val = float(algo_cfg.get("critic_lr", args.critic_lr if args.critic_lr is not None else args.lr))
    # Use initial_noise_std for agent initialization (will be updated during training)
    exploration_std_val = float(exploration_cfg.get("initial_noise_std", args.exploration_std)) if exploration_cfg else (args.exploration_std if args.exploration_std else 1.0)
    target_std_val = float(algo_cfg.get("target_noise", args.target_std)) if algo_cfg else args.target_std
    noise_clip_val = float(algo_cfg.get("noise_clip", args.noise_clip)) if algo_cfg else args.noise_clip

    config = TD3Config(
        gamma=gamma_val,
        tau=tau_val,
        policy_delay=args.policy_delay,
        lr=args.lr,
        actor_lr=actor_lr_val,
        critic_lr=critic_lr_val,
        exploration_std=exploration_std_val,
        target_std=target_std_val,
        noise_clip=noise_clip_val,
        action_low=float(env.action_space.low.min()),
        action_high=float(env.action_space.high.max()),
    )

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        device=device,
    )

    # Optional learning-rate scheduler from YAML
    actor_scheduler = None
    critic_scheduler = None
    lr_sched_cfg = train_cfg.get("lr_schedule") if isinstance(train_cfg, dict) else None
    if isinstance(lr_sched_cfg, dict) and lr_sched_cfg.get("type", "").lower() == "step":
        from torch.optim.lr_scheduler import StepLR
        decay_factor = float(lr_sched_cfg.get("decay_factor", 0.5))
        decay_epochs = int(lr_sched_cfg.get("decay_epochs", 50))
        actor_scheduler = StepLR(agent.actor_optimizer, step_size=decay_epochs, gamma=decay_factor)
        critic_scheduler = StepLR(agent.critic_optimizer, step_size=decay_epochs, gamma=decay_factor)
        print(f"LR schedule enabled: StepLR every {decay_epochs} epochs, factor={decay_factor}")

    # Create buffer with matching n_step and gamma from TD3 config
    # This ensures n-step returns (if used) are computed with the correct discount factor
    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=args.buffer_size,
        n_step=config.n_step,  # Match TD3 config
        gamma=config.gamma,  # Match TD3 config
    )

    total_episodes = args.epochs * args.days_per_epoch

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_writer = initialize_csv(args.csv_log) if args.csv_log else None

    training_metrics_path = None
    if args.quick_eval_outdir:
        args.quick_eval_outdir.mkdir(parents=True, exist_ok=True)
        training_metrics_path = args.quick_eval_outdir / "training_metrics.jsonl"
        try:
            training_metrics_path.write_text("", encoding="utf-8")
        except Exception:
            pass

    episode_returns: List[float] = []
    print(f"Using device={device}, seed={args.seed}")
    
    # Calculate total episodes for noise decay
    total_episodes = args.epochs * args.days_per_epoch
    print(f"Starting training on device {device} with {total_episodes} episodes")
    print(f"Gaussian exploration noise: {initial_noise_std:.2f} -> {final_noise_std:.2f} over {noise_decay_episodes} episodes")
    
    def get_noise_std(episode: int) -> float:
        """Calculate exploration noise std with linear decay."""
        if episode < args.warmup_steps // 24:  # During warmup, use high noise
            return initial_noise_std
        progress = min(1.0, max(0.0, (episode - args.warmup_steps // 24) / noise_decay_episodes))
        return initial_noise_std - (initial_noise_std - final_noise_std) * progress

    episode_counter = 0
    last_avg_return = float("nan")
    warmup_complete_logged = False
    
    # Track action diversity during warmup
    warmup_actions = {
        'charge': 0,      # actions < -0.1
        'idle': 0,        # -0.1 <= action <= 0.1
        'discharge': 0,  # actions > 0.1
        'total': 0,
        'min_action': float('inf'),
        'max_action': float('-inf')
    }

    try:
        for epoch in range(args.epochs):
            for day in range(args.days_per_epoch):
                episode_counter += 1
                episode_seed = int(env.np_random.integers(0, 2**32 - 1, dtype=np.uint32))
                day_index = episode_seed % len(env.dates)
                selected_day = env.dates[int(day_index)]
                obs, _ = env.reset(seed=episode_seed, options={"date": selected_day})
                obs_vec = np.asarray(obs, dtype=np.float32)

                done = False
                episode_return = 0.0
                metrics_accumulator: Dict[str, float] = {
                    "revenue_pv_grid": 0.0,
                    "revenue_energy": 0.0,
                    "revenue_reserve": 0.0,
                    "cost_degradation": 0.0,
                }
                step_info: Dict[str, float] = {}
                episode_update_stats: List[Dict[str, float]] = []

                step = 0
                # Maintain full observation history within the episode
                obs_history: List[np.ndarray] = []
                # Reset hidden state at episode start
                agent.reset_hidden_state()
                while not done:
                    step += 1
                    # RL clean-up patch: append and truncate to last seq_len steps
                    obs_history.append(obs_vec)
                    if len(obs_history) > args.seq_len:
                        obs_history.pop(0)
                    if len(buffer) < args.warmup_steps:
                        # Warmup: guided exploration (70% rule-based hints + 30% random)
                        action = guided_warmup_action(env, obs_vec, random_prob=0.3)
                        
                        # Track action diversity
                        warmup_actions['total'] += 1
                        action_val = float(action[0])
                        warmup_actions['min_action'] = min(warmup_actions['min_action'], action_val)
                        warmup_actions['max_action'] = max(warmup_actions['max_action'], action_val)
                        if action_val < -0.1:
                            warmup_actions['charge'] += 1
                        elif action_val > 0.1:
                            warmup_actions['discharge'] += 1
                        else:
                            warmup_actions['idle'] += 1
                    else:
                        # After warmup: Gaussian exploration with decay
                        # Update agent's exploration std based on episode
                        current_noise_std = get_noise_std(episode_counter)
                        agent.config.exploration_std = current_noise_std
                        
                        # Policy action with Gaussian noise (noise added in agent.act)
                        state_seq = np.asarray(obs_history, dtype=np.float32)[np.newaxis, :, :]  # [1, T, F]
                        # Hidden state is already reset at episode start, no need to reset again
                        action_vec = agent.act(
                            state_seq,
                            eval_mode=args.eval_mode,
                            reset_hidden=False,
                        )[0]
                        action = np.asarray(action_vec, dtype=np.float32)

                    next_obs, reward, terminated, truncated, step_info = env.step(action)

                    done = terminated or truncated
                    next_obs_vec = np.asarray(next_obs, dtype=np.float32)

                    buffer.add(obs_vec, action, reward, next_obs_vec, done)
                    episode_return += float(reward)
                    metrics_accumulator["revenue_pv_grid"] += step_info.get("revenue_pv_grid", 0.0)
                    metrics_accumulator["revenue_energy"] += step_info.get("revenue_energy", 0.0)
                    metrics_accumulator["revenue_reserve"] += step_info.get("revenue_reserve", 0.0)
                    metrics_accumulator["cost_degradation"] += step_info.get("cost_degradation", 0.0)

                    obs_vec = next_obs_vec

                    if len(buffer) >= args.warmup_steps:
                        # Log warmup summary once
                        if not warmup_complete_logged and warmup_actions['total'] > 0:
                            warmup_complete_logged = True
                            total = warmup_actions['total']
                            charge_pct = 100.0 * warmup_actions['charge'] / total
                            idle_pct = 100.0 * warmup_actions['idle'] / total
                            discharge_pct = 100.0 * warmup_actions['discharge'] / total
                            action_range = warmup_actions['max_action'] - warmup_actions['min_action']
                            print(f"\n[Warmup Complete] Buffer size: {len(buffer)}")
                            print(f"  Action distribution: Charge={charge_pct:.1f}%, Idle={idle_pct:.1f}%, Discharge={discharge_pct:.1f}%")
                            print(f"  Action range: [{warmup_actions['min_action']:.3f}, {warmup_actions['max_action']:.3f}] (span={action_range:.3f})")
                            if action_range < 1.5:
                                print(f"  WARNING: Action space coverage may be insufficient!")
                            print()
                        
                        for _ in range(args.updates_per_step):
                            stats = agent.update(buffer, batch_size=args.batch_size, seq_len=args.seq_len)
                            episode_update_stats.append(stats)

                # Episode finished
                episode_returns.append(episode_return)
                episode_profit = step_info.get("episode_profit", episode_return)
                episode_revenue = step_info.get("episode_revenue", metrics_accumulator["revenue_pv_grid"] + metrics_accumulator["revenue_energy"] + metrics_accumulator["revenue_reserve"])
                episode_degradation = step_info.get("episode_degradation", metrics_accumulator["cost_degradation"])

                if training_metrics_path is not None:
                    def _avg(key: str) -> float:
                        vals = [float(s.get(key, float("nan"))) for s in episode_update_stats if key in s]
                        return float(sum(vals) / len(vals)) if vals else float("nan")

                    metrics_entry = {
                        "episode": episode_counter,
                        "critic_loss": _avg("critic_loss"),
                        "actor_loss": _avg("actor_loss"),
                        "q1_mean": _avg("q1_mean"),
                        "q2_mean": _avg("q2_mean"),
                        "q1_mean_all": _avg("q1_mean_all"),
                        "q2_mean_all": _avg("q2_mean_all"),
                    }
                    try:
                        with training_metrics_path.open("a", encoding="utf-8") as tm_file:
                            json.dump(metrics_entry, tm_file)
                            tm_file.write("\n")
                    except Exception:
                        pass

                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            episode_counter,
                            episode_return,
                            episode_profit,
                            episode_revenue,
                            episode_degradation,
                            metrics_accumulator["revenue_pv_grid"],
                            metrics_accumulator["revenue_energy"],
                            metrics_accumulator["revenue_reserve"],
                            metrics_accumulator["cost_degradation"],
                        ]
                    )
                    csv_writer.file.flush()

                if episode_counter % args.log_interval == 0:
                    recent_returns = episode_returns[-args.log_interval :]
                    avg_return = np.mean(recent_returns)
                    last_avg_return = float(avg_return)
                    print(
                        f"Episode {episode_counter:04d} | Epoch {epoch+1}/{args.epochs} | Return {episode_return:8.2f} | "
                        f"Avg{args.log_interval}: {avg_return:8.2f} | Profit {episode_profit:8.2f} | "
                        f"Revenue {episode_revenue:8.2f} | Degradation {episode_degradation:8.2f}"
                    )

                if episode_counter % args.checkpoint_interval == 0:
                    if not np.isfinite(last_avg_return):
                        window = episode_returns[-args.log_interval :]
                        if window:
                            last_avg_return = float(np.mean(window))
                        elif episode_returns:
                            last_avg_return = float(np.mean(episode_returns))
                        else:
                            last_avg_return = float("nan")

                    actor_path = args.checkpoint_dir / f"actor_ep{episode_counter:04d}.pth"
                    critic_path = args.checkpoint_dir / f"critic_ep{episode_counter:04d}.pth"
                    torch.save(agent.actor.state_dict(), actor_path)
                    torch.save(agent.critic.state_dict(), critic_path)
                    meta_path = args.checkpoint_dir / "checkpoint_meta.json"
                    meta_path.parent.mkdir(parents=True, exist_ok=True)
                    with meta_path.open("a", encoding="utf-8") as meta_file:
                        json.dump({"episode": episode_counter, "avg_return": last_avg_return}, meta_file)
                        meta_file.write("\n")
                    print(f"Saved checkpoints at episode {episode_counter} -> {actor_path}, {critic_path}")

                    # Quick 1-day rollout (optional)
                    if args.quick_eval_interval and args.quick_eval_interval > 0:
                        if episode_counter % int(args.quick_eval_interval) == 0:
                            if not args.quick_eval_date:
                                print("quick-eval skipped: --quick-eval-date not set")
                            else:
                                try:
                                    qe_env = BESSEnv(data_path=args.data_path, degradation_model=args.degradation_model)
                                    rows = quick_rollout(qe_env, agent, args.quick_eval_date)
                                    args.quick_eval_outdir.mkdir(parents=True, exist_ok=True)
                                    out_csv = args.quick_eval_outdir / f"hourly_ep{episode_counter:04d}.csv"
                                    with out_csv.open("w", newline="") as f:
                                        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                                        writer.writeheader()
                                        writer.writerows(rows)
                                    print(f"[quick] saved {out_csv}")
                                except Exception as e:
                                    print(f"[quick] rollout failed: {e}")
            # Step LR schedulers at end of epoch
            if actor_scheduler is not None:
                actor_scheduler.step()
                critic_scheduler.step()
                if (epoch + 1) % max(1, int(lr_sched_cfg.get("decay_epochs", 50))) == 0:
                    cur_actor_lr = agent.actor_optimizer.param_groups[0]["lr"]
                    cur_critic_lr = agent.critic_optimizer.param_groups[0]["lr"]
                    print(f"[LR] Epoch {epoch+1}: actor_lr={cur_actor_lr:.6g}, critic_lr={cur_critic_lr:.6g}")

        print("Training completed")
    except KeyboardInterrupt:
        print("KeyboardInterrupt received; terminating training loop.")
    finally:
        close_csv(csv_writer)
        print("Training interrupted, CSV closed safely")


if __name__ == "__main__":
    main()



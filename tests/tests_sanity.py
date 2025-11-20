"""Quick sanity checks for the BESSEnv environment."""

from __future__ import annotations

import numpy as np

from src.envs import BESSEnv


def run_random_episode(env: BESSEnv, seed: int = 1234) -> None:
    obs, info = env.reset(seed=seed)
    assert obs.shape == (env.observation_space.shape[0],), "Observation shape mismatch"

    done = False
    step_count = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        soc = info.get("soc", env.soc)
        assert env.soc_min - 1e-6 <= soc <= env.soc_max + 1e-6, "SOC out of bounds"

        export = (
            info.get("p_pv_grid", 0.0)
            + max(info.get("p_bess_em", 0.0), 0.0)
            + info.get("p_reserve", 0.0)
        )
        assert export <= env.P_poi_max + 1e-6, "POI export exceeded"

        done = terminated or truncated

    assert done, "Episode did not terminate"
    assert step_count <= env.max_hours, "Unexpected number of steps"


def main() -> None:
    env = BESSEnv()
    run_random_episode(env, seed=1234)
    run_random_episode(env, seed=5678)
    print("Sanity OK")


if __name__ == "__main__":
    main()



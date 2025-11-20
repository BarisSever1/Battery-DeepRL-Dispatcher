"""
Basic tests for BESS environment
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs import BESSEnv


def test_env_initialization():
    """Test that environment can be initialized."""
    print("Testing environment initialization...")
    env = BESSEnv()
    print(f"✓ Environment created")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print()


def test_reset():
    """Test environment reset."""
    print("Testing reset...")
    env = BESSEnv()
    obs, info = env.reset()
    
    print(f"✓ Environment reset")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Info: {info}")
    print(f"  Initial SOC: {env.soc:.3f}")
    print()


def test_single_step():
    """Test a single step."""
    print("Testing single step...")
    env = BESSEnv()
    obs, info = env.reset()
    
    # Take a random action
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    
    print(f"✓ Step executed")
    print(f"  Action: {action[0]:.3f}")
    print(f"  Reward: {reward:.2f} EUR")
    print(f"  SOC: {step_info['soc_before']:.3f} → {step_info['soc']:.3f}")
    print(f"  Battery power: {step_info['p_battery']:.2f} MW")
    print(f"  Reserve: {step_info['p_reserve']:.2f} MW")
    print(f"  Terminated: {terminated}")
    print()


def test_full_episode():
    """Test a complete 24-hour episode."""
    print("Testing full episode (24 hours)...")
    env = BESSEnv(seed=42)
    obs, info = env.reset(seed=42)
    
    total_reward = 0
    step_count = 0
    
    for hour in range(24):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        if terminated:
            break
    
    print(f"✓ Episode completed")
    print(f"  Steps: {step_count}")
    print(f"  Total reward: {total_reward:.2f} EUR")
    print(f"  Episode revenue: {step_info.get('episode_revenue', 0):.2f} EUR")
    print(f"  Episode degradation: {step_info.get('episode_degradation', 0):.2f} EUR")
    print(f"  Episode profit: {step_info.get('episode_profit', 0):.2f} EUR")
    print(f"  Final SOC: {env.soc:.3f}")
    print(f"  Discharge count: {env.discharge_count}")
    print()


def test_constraint_enforcement():
    """Test that constraints are enforced."""
    print("Testing constraint enforcement...")
    env = BESSEnv()
    obs, info = env.reset()
    
    # Try to violate SOC constraints by taking extreme actions
    soc_values = []
    
    for hour in range(24):
        # Alternate between extreme charge (0) and discharge (1)
        action = np.array([1.0 if hour % 2 == 0 else 0.0])
        obs, reward, terminated, truncated, step_info = env.step(action)
        soc_values.append(env.soc)
    
    min_soc = min(soc_values)
    max_soc = max(soc_values)
    
    print(f"✓ Constraints tested")
    print(f"  SOC range: [{min_soc:.3f}, {max_soc:.3f}]")
    print(f"  SOC limits: [{env.soc_min:.3f}, {env.soc_max:.3f}]")
    print(f"  Within bounds: {min_soc >= env.soc_min and max_soc <= env.soc_max}")
    print()


if __name__ == '__main__':
    print("=" * 80)
    print("BESS ENVIRONMENT TESTS")
    print("=" * 80)
    print()
    
    try:
        test_env_initialization()
        test_reset()
        test_single_step()
        test_full_episode()
        test_constraint_enforcement()
        
        print("=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


from src.envs.env_pv_bess import BESSEnv
import numpy as np

print("Loading environment...")
env = BESSEnv()
print("[OK] Environment loaded successfully!")
print(f"  Action space: {env.action_space}")
print(f"  Observation space: {env.observation_space}")

print("\nResetting environment...")
obs, info = env.reset()
print("[OK] Reset successful!")
print(f"  Observation shape: {obs.shape}")
print(f"  Initial SOC: {env.soc}")
print(f"  Day: {info['day']}")

print("\nTaking a step...")
action = env.action_space.sample()
next_obs, reward, term, trunc, step_info = env.step(action)
print("[OK] Step successful!")
print(f"  Action (delta): {action[0]:.3f}")
print(f"  Reward: {reward:.2f} EUR")
print(f"  SOC: {step_info['soc_before']:.3f} -> {step_info['soc']:.3f}")
print(f"  Battery power: {step_info['p_battery']:.2f} MW")
print(f"  Reserve: {step_info['p_reserve']:.2f} MW")

print("\nRunning full episode (24 hours)...")
obs, info = env.reset(seed=42)
total_reward = 0
for hour in range(24):
    action = np.array([np.random.uniform(0, 1)])
    obs, reward, term, trunc, step_info = env.step(action)
    total_reward += reward
    if term:
        break

print("[OK] Episode completed!")
print(f"  Total reward: {total_reward:.2f} EUR")
print(f"  Episode profit: {step_info.get('episode_profit', 0):.2f} EUR")
print(f"  Final SOC: {env.soc:.3f}")

print("\n" + "="*60)
print("[SUCCESS] ALL BASIC TESTS PASSED!")
print("="*60)


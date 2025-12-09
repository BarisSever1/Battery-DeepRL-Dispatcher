"""
Test script to evaluate reward scenarios with the new arbitrage-sensitive reward shaping.
Tests various scenarios organized by category to understand reward magnitudes and net profitability signals.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs import BESSEnv


def test_scenario(env, scenario_name, hour, action, soc_start, description, setup_actions=None):
    """
    Test a single scenario and return detailed reward information.
    
    Args:
        env: BESSEnv instance
        scenario_name: Name of the scenario
        hour: Target hour to test (0-23)
        action: Action to take at target hour
        soc_start: Desired SOC at the start of target hour
        description: Description of the scenario
        setup_actions: Optional list of (hour, action) tuples to execute before target hour
                      to set up DOD state or SOC
    """
    # Reset environment to the same date as quick eval (2024-01-08)
    import numpy as np
    eval_date = np.datetime64('2024-01-08').astype('datetime64[D]').astype(object)
    env.reset(seed=42, options={'date': eval_date})
    
    # Step through environment to reach target hour, executing setup actions if provided
    # This ensures DOD tracking and other state is correct
    # We step from hour 0 to hour-1, so the last step takes us to the target hour
    for h in range(hour):
        # Check if there's a setup action for this hour
        setup_action = None
        if setup_actions:
            for setup_h, setup_a in setup_actions:
                if setup_h == h:
                    setup_action = setup_a
                    break
        
        # On the last step (h == hour - 1), adjust SOC if needed and no setup action conflicts
        if h == hour - 1 and setup_action is None:
            # Try to adjust SOC on the step that takes us to target hour
            current_soc = env.soc
            soc_diff = soc_start - current_soc
            if abs(soc_diff) > 0.02:  # If difference is significant
                # Calculate action needed (accounting for efficiency)
                # Rough estimate: action of -1.0 charges about 0.33 SOC, +1.0 discharges about 0.30 SOC
                if soc_diff > 0:
                    # Need to charge
                    adjust_action = -min(1.0, abs(soc_diff) / 0.33)
                else:
                    # Need to discharge
                    adjust_action = min(1.0, abs(soc_diff) / 0.30)
                setup_action = adjust_action
        
        # Execute action (setup action, SOC adjustment, or idle)
        action_to_take = setup_action if setup_action is not None else 0.0
        obs, _, _, _, _ = env.step(np.array([action_to_take], dtype=np.float32))
        
        # Safety check: if we've overshot, something went wrong
        if env.current_hour > hour:
            print(f"Warning: Overshot target hour {hour}, now at {env.current_hour}")
            break
    
    # Verify we're at the correct hour
    if env.current_hour != hour:
        print(f"Error: Expected hour {hour}, but environment is at hour {env.current_hour}. Cannot proceed.")
        # Return a dummy result indicating failure
        return {
            'scenario': scenario_name,
            'description': description + " [FAILED: Wrong hour]",
            'hour': env.current_hour,
            'action': action,
            'soc_start': env.soc,
            'soc_after': env.soc,
            'p_battery': 0.0,
            'energy_mwh': 0.0,
            'price_em': 0.0,
            'price_as': 0.0,
            'is_peak': False,
            'is_cheap': False,
            'next_peak_hour': None,
            'next_peak_price': None,
            'previous_cheap_price': None,
            'cost_degradation': 0.0,
            'degradation_per_mwh': 0.0,
            'arbitrage_per_mwh': 0.0,
            'net_profit_per_mwh': 0.0,
            'net_profit_scaled': 0.0,
            'reward': -1000.0,  # Large penalty for test failure
            'dod_current': 0.0,
        }
    
    # Get current state info (at target hour)
    row = env.day_data.iloc[env.current_hour]
    price_em = env._denormalize('price_em', row['price_em'])
    price_as = env._denormalize('price_as', row['price_as'])
    p_pv = env._denormalize('p_res_total', row['p_res_total'])
    
    # Get peak/cheap hour info
    is_peak = env.current_hour in getattr(env, 'peak_hours', set())
    is_cheap = env.current_hour in getattr(env, 'cheapest_hours', set())
    
    # Find next peak hour (using same logic as environment)
    next_peak_hour = None
    peak_hours = getattr(env, 'peak_hours', set())
    if peak_hours:
        if env.current_hour in peak_hours:
            next_peak_hour = env.current_hour
        else:
            for h in sorted(peak_hours):
                if h > env.current_hour:
                    next_peak_hour = h
                    break
    
    next_peak_price = None
    if next_peak_hour is not None and 0 <= next_peak_hour < 24:
        try:
            next_peak_price = env._denormalize('price_em', env.day_data.iloc[next_peak_hour]['price_em'])
        except:
            pass
    
    # Get previous cheap price
    previous_cheap_price = None
    if hasattr(env, 'cheapest_hours') and env.cheapest_hours:
        try:
            cheap_prices = []
            for h in env.cheapest_hours:
                if h < env.current_hour and 0 <= h < 24:
                    try:
                        p = env._denormalize('price_em', env.day_data.iloc[h]['price_em'])
                        if p > 0:  # Only valid prices
                            cheap_prices.append(p)
                    except:
                        continue
            if cheap_prices:
                previous_cheap_price = min(cheap_prices)
        except:
            pass
    
    # Execute the test action
    obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
    
    # Extract key metrics
    p_battery = info.get('p_battery', 0.0)
    energy_mwh = info.get('energy_mwh', 0.0)
    cost_degradation = info.get('cost_degradation', 0.0)
    # Note: These fields were removed when we simplified reward shaping
    # They're kept for compatibility with the test script but always return 0.0
    degradation_per_mwh = 0.0
    arbitrage_per_mwh = 0.0
    net_profit_per_mwh = 0.0
    net_profit_scaled = info.get('net_profit_scaled', 0.0)
    soc_after = info.get('soc', env.soc)
    dod_current = info.get('dod_morning', 0.0) if env.current_hour < 12 else info.get('dod_evening', 0.0)
    
    return {
        'scenario': scenario_name,
        'description': description,
        'hour': env.current_hour,  # Use actual hour from environment
        'action': action,
        'soc_start': env.soc_previous if hasattr(env, 'soc_previous') else soc_start,
        'soc_after': soc_after,
        'p_battery': p_battery,
        'energy_mwh': energy_mwh,
        'price_em': price_em,
        'price_as': price_as,
        'is_peak': is_peak,
        'is_cheap': is_cheap,
        'next_peak_hour': next_peak_hour,
        'next_peak_price': next_peak_price,
        'previous_cheap_price': previous_cheap_price,
        'cost_degradation': cost_degradation,
        'degradation_per_mwh': degradation_per_mwh,
        'arbitrage_per_mwh': arbitrage_per_mwh,
        'net_profit_per_mwh': net_profit_per_mwh,
        'net_profit_scaled': net_profit_scaled,
        'reward': reward,
        'dod_current': dod_current,
    }


def print_scenario_group(group_name, results_group):
    """Print a group of related scenarios."""
    print(f"\n{'='*80}")
    print(f"{group_name.upper()}")
    print(f"{'='*80}")
    for result in results_group:
        print(f"\n{result['scenario']}: {result['description']}")
        print(f"  Reward: {result['reward']:.2f}")
        print(f"  Arbitrage per MWh: {result['arbitrage_per_mwh']:.2f} EUR/MWh")
        print(f"  Degradation per MWh: {result['degradation_per_mwh']:.2f} EUR/MWh")
        print(f"  Net Profit per MWh: {result['net_profit_per_mwh']:.2f} EUR/MWh")
        if result['dod_current'] > 0:
            print(f"  DOD: {result['dod_current']:.3f}")
        if result['energy_mwh'] > 0:
            print(f"  Energy: {result['energy_mwh']:.2f} MWh")


def run_all_scenarios():
    """Run comprehensive reward scenario tests organized by category."""
    print("=" * 80)
    print("COMPREHENSIVE REWARD SCENARIO TESTING")
    print("Using date: 2024-01-08 (same as quick eval)")
    print("=" * 80)
    print()
    
    # Initialize environment
    env = BESSEnv()
    
    all_results = []
    
    # ============================================================================
    # GROUP 1: DISCHARGE AT PEAK HOURS (Varying Arbitrage & DOD)
    # ============================================================================
    group1_results = []
    
    # Low DOD scenario (shallow cycle) - charge a bit at hour 2, then discharge at hour 8
    result = test_scenario(
        env, "Discharge_Peak_LowDOD_LowSOC",
        hour=8, action=0.5, soc_start=0.6,
        description="Discharge 50% at peak (hour 8), low DOD, low SOC (0.6)",
        setup_actions=[(2, -0.3)]  # Small charge at hour 2 to create low DOD
    )
    group1_results.append(result)
    all_results.append(result)
    
    # Medium DOD scenario - charge more at hour 2
    result = test_scenario(
        env, "Discharge_Peak_MedDOD_MedSOC",
        hour=8, action=0.7, soc_start=0.7,
        description="Discharge 70% at peak (hour 8), medium DOD, medium SOC (0.7)",
        setup_actions=[(2, -0.7)]  # Medium charge at hour 2
    )
    group1_results.append(result)
    all_results.append(result)
    
    # High DOD scenario (deep cycle) - full charge at hour 2
    result = test_scenario(
        env, "Discharge_Peak_HighDOD_HighSOC",
        hour=8, action=1.0, soc_start=0.95,
        description="Discharge 100% at peak (hour 8), high DOD, high SOC (0.95)",
        setup_actions=[(2, -1.0)]  # Full charge at hour 2 to create high DOD
    )
    group1_results.append(result)
    all_results.append(result)
    
    # Low SOC at peak (no previous charge, low arbitrage)
    result = test_scenario(
        env, "Discharge_Peak_LowSOC_NoCharge",
        hour=8, action=0.8, soc_start=0.3,
        description="Discharge 80% at peak (hour 8), low SOC (0.3), no previous charge"
    )
    group1_results.append(result)
    all_results.append(result)
    
    # Very high SOC at peak (minimal discharge possible)
    result = test_scenario(
        env, "Discharge_Peak_VeryHighSOC",
        hour=8, action=0.2, soc_start=0.98,
        description="Discharge 20% at peak (hour 8), very high SOC (0.98)",
        setup_actions=[(2, -1.0)]  # Charge to high SOC first
    )
    group1_results.append(result)
    all_results.append(result)
    
    # Small discharge at peak (shallow, low degradation)
    result = test_scenario(
        env, "Discharge_Peak_SmallAction",
        hour=8, action=0.1, soc_start=0.6,
        description="Small discharge 10% at peak (hour 8), mid SOC (0.6)"
    )
    group1_results.append(result)
    all_results.append(result)
    
    print_scenario_group("GROUP 1: DISCHARGE AT PEAK HOURS", group1_results)
    
    # ============================================================================
    # GROUP 2: CHARGE AT CHEAP HOURS (Varying Arbitrage & DOD)
    # ============================================================================
    group2_results = []
    
    # Low SOC, low DOD (shallow charge)
    result = test_scenario(
        env, "Charge_Cheap_LowSOC_LowDOD",
        hour=2, action=-0.3, soc_start=0.2,
        description="Charge 30% at cheap hour (hour 2), low SOC (0.2), low DOD"
    )
    group2_results.append(result)
    all_results.append(result)
    
    # Low SOC, high DOD (deep charge from very low)
    result = test_scenario(
        env, "Charge_Cheap_LowSOC_HighDOD",
        hour=2, action=-0.8, soc_start=0.3,
        description="Charge 80% at cheap hour (hour 2), low SOC (0.3), high DOD",
        setup_actions=[(0, -0.5)]  # Partial charge at hour 0 to create high DOD
    )
    group2_results.append(result)
    all_results.append(result)
    
    # Medium SOC, low DOD
    result = test_scenario(
        env, "Charge_Cheap_MedSOC_LowDOD",
        hour=2, action=-0.5, soc_start=0.5,
        description="Charge 50% at cheap hour (hour 2), medium SOC (0.5), low DOD"
    )
    group2_results.append(result)
    all_results.append(result)
    
    # High SOC, low DOD (approaching max)
    result = test_scenario(
        env, "Charge_Cheap_HighSOC_LowDOD",
        hour=2, action=-0.3, soc_start=0.85,
        description="Charge 30% at cheap hour (hour 2), high SOC (0.85), low DOD"
    )
    group2_results.append(result)
    all_results.append(result)
    
    # Very high SOC (should be penalized)
    result = test_scenario(
        env, "Charge_Cheap_VeryHighSOC",
        hour=2, action=-0.5, soc_start=0.95,
        description="Charge 50% at cheap hour (hour 2), very high SOC (0.95) - should be penalized"
    )
    group2_results.append(result)
    all_results.append(result)
    
    # Full charge at cheap hour
    result = test_scenario(
        env, "Charge_Cheap_FullCharge",
        hour=2, action=-1.0, soc_start=0.2,
        description="Full charge at cheap hour (hour 2), low SOC (0.2)"
    )
    group2_results.append(result)
    all_results.append(result)
    
    # Small charge at cheap hour
    result = test_scenario(
        env, "Charge_Cheap_SmallAction",
        hour=2, action=-0.1, soc_start=0.5,
        description="Small charge 10% at cheap hour (hour 2), mid SOC (0.5)"
    )
    group2_results.append(result)
    all_results.append(result)
    
    print_scenario_group("GROUP 2: CHARGE AT CHEAP HOURS", group2_results)
    
    # ============================================================================
    # GROUP 3: BETWEEN PEAKS (Charge/Discharge/Idle)
    # ============================================================================
    group3_results = []
    
    # Charge between peaks (preparing for evening peak)
    result = test_scenario(
        env, "Charge_BetweenPeaks_LowSOC",
        hour=12, action=-0.6, soc_start=0.25,
        description="Charge 60% between peaks (hour 12), low SOC (0.25), preparing for evening peak"
    )
    group3_results.append(result)
    all_results.append(result)
    
    # Charge between peaks (medium SOC)
    result = test_scenario(
        env, "Charge_BetweenPeaks_MedSOC",
        hour=12, action=-0.4, soc_start=0.4,
        description="Charge 40% between peaks (hour 12), medium SOC (0.4)"
    )
    group3_results.append(result)
    all_results.append(result)
    
    # Discharge between peaks (unusual, but test it)
    result = test_scenario(
        env, "Discharge_BetweenPeaks",
        hour=12, action=0.5, soc_start=0.7,
        description="Discharge 50% between peaks (hour 12), high SOC (0.7)"
    )
    group3_results.append(result)
    all_results.append(result)
    
    # Idle between peaks (smart waiting)
    result = test_scenario(
        env, "Idle_BetweenPeaks_GoodSOC",
        hour=12, action=0.0, soc_start=0.5,
        description="Idle between peaks (hour 12), good SOC (0.5) - smart waiting"
    )
    group3_results.append(result)
    all_results.append(result)
    
    # Idle between peaks (low SOC - should prepare)
    result = test_scenario(
        env, "Idle_BetweenPeaks_LowSOC",
        hour=12, action=0.0, soc_start=0.2,
        description="Idle between peaks (hour 12), low SOC (0.2) - missed opportunity"
    )
    group3_results.append(result)
    all_results.append(result)
    
    # Idle between peaks (high SOC - OK to wait)
    result = test_scenario(
        env, "Idle_BetweenPeaks_HighSOC",
        hour=12, action=0.0, soc_start=0.8,
        description="Idle between peaks (hour 12), high SOC (0.8) - OK to wait"
    )
    group3_results.append(result)
    all_results.append(result)
    
    print_scenario_group("GROUP 3: BETWEEN PEAKS", group3_results)
    
    # ============================================================================
    # GROUP 4: IDLING SCENARIOS
    # ============================================================================
    group4_results = []
    
    # Idle at peak with high SOC (worst case)
    result = test_scenario(
        env, "Idle_Peak_HighSOC",
        hour=8, action=0.0, soc_start=0.9,
        description="Idle at peak (hour 8), high SOC (0.9) - worst case, missed opportunity"
    )
    group4_results.append(result)
    all_results.append(result)
    
    # Idle at peak with low SOC (can't discharge anyway)
    result = test_scenario(
        env, "Idle_Peak_LowSOC",
        hour=8, action=0.0, soc_start=0.1,
        description="Idle at peak (hour 8), low SOC (0.1) - can't discharge anyway"
    )
    group4_results.append(result)
    all_results.append(result)
    
    # Idle at cheap hour with low SOC (missed opportunity)
    result = test_scenario(
        env, "Idle_Cheap_LowSOC",
        hour=2, action=0.0, soc_start=0.2,
        description="Idle at cheap hour (hour 2), low SOC (0.2) - missed opportunity"
    )
    group4_results.append(result)
    all_results.append(result)
    
    # Idle at cheap hour with high SOC (OK)
    result = test_scenario(
        env, "Idle_Cheap_HighSOC",
        hour=2, action=0.0, soc_start=0.9,
        description="Idle at cheap hour (hour 2), high SOC (0.9) - OK, already charged"
    )
    group4_results.append(result)
    all_results.append(result)
    
    # Idle at normal hour (smart preservation)
    result = test_scenario(
        env, "Idle_NormalHour",
        hour=10, action=0.0, soc_start=0.6,
        description="Idle at normal hour (hour 10), mid SOC (0.6) - smart preservation"
    )
    group4_results.append(result)
    all_results.append(result)
    
    print_scenario_group("GROUP 4: IDLING SCENARIOS", group4_results)
    
    # ============================================================================
    # GROUP 5: MISALIGNED ACTIONS (Wrong timing)
    # ============================================================================
    group5_results = []
    
    # Charge at peak hour (wrong)
    result = test_scenario(
        env, "Charge_PeakHour",
        hour=8, action=-0.8, soc_start=0.5,
        description="Charge 80% at peak hour (hour 8) - wrong timing"
    )
    group5_results.append(result)
    all_results.append(result)
    
    # Discharge at cheap hour (wrong)
    result = test_scenario(
        env, "Discharge_CheapHour",
        hour=2, action=0.8, soc_start=0.7,
        description="Discharge 80% at cheap hour (hour 2) - wrong timing"
    )
    group5_results.append(result)
    all_results.append(result)
    
    # Charge at normal hour (not optimal)
    result = test_scenario(
        env, "Charge_NormalHour",
        hour=10, action=-0.6, soc_start=0.4,
        description="Charge 60% at normal hour (hour 10) - not optimal"
    )
    group5_results.append(result)
    all_results.append(result)
    
    # Discharge at normal hour (not optimal)
    result = test_scenario(
        env, "Discharge_NormalHour",
        hour=14, action=0.6, soc_start=0.7,
        description="Discharge 60% at normal hour (hour 14) - not optimal"
    )
    group5_results.append(result)
    all_results.append(result)
    
    print_scenario_group("GROUP 5: MISALIGNED ACTIONS", group5_results)
    
    # ============================================================================
    # GROUP 6: EXTREME SCENARIOS
    # ============================================================================
    group6_results = []
    
    # Maximum charge from minimum SOC
    result = test_scenario(
        env, "Charge_MaxFromMin",
        hour=2, action=-1.0, soc_start=0.05,
        description="Maximum charge from minimum SOC (0.05) at cheap hour"
    )
    group6_results.append(result)
    all_results.append(result)
    
    # Maximum discharge from maximum SOC
    result = test_scenario(
        env, "Discharge_MaxFromMax",
        hour=8, action=1.0, soc_start=0.95,
        description="Maximum discharge from maximum SOC (0.95) at peak hour"
    )
    group6_results.append(result)
    all_results.append(result)
    
    # Very small action (minimal cycling)
    result = test_scenario(
        env, "Action_VerySmall",
        hour=8, action=0.05, soc_start=0.6,
        description="Very small action (5%) at peak hour - minimal cycling"
    )
    group6_results.append(result)
    all_results.append(result)
    
    # Charge when already at max (should be penalized)
    result = test_scenario(
        env, "Charge_AtMaxSOC",
        hour=2, action=-0.5, soc_start=0.99,
        description="Charge when already at max SOC (0.99) - should be penalized"
    )
    group6_results.append(result)
    all_results.append(result)
    
    # Discharge when already at min (should be penalized)
    result = test_scenario(
        env, "Discharge_AtMinSOC",
        hour=8, action=0.5, soc_start=0.01,
        description="Discharge when already at min SOC (0.01) - should be penalized"
    )
    group6_results.append(result)
    all_results.append(result)
    
    print_scenario_group("GROUP 6: EXTREME SCENARIOS", group6_results)
    
    # ============================================================================
    # SUMMARY AND ANALYSIS
    # ============================================================================
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    import sys
    output_name = "reward_scenarios_test.csv"
    if len(sys.argv) > 1:
        output_name = sys.argv[1]
        if not output_name.endswith('.csv'):
            output_name += '.csv'
    
    output_path = project_root / "results" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    except PermissionError:
        print(f"\nWarning: Could not save CSV (file may be open). Results displayed above.")
        # Try alternative filename
        alt_path = project_root / "results" / f"{output_name.replace('.csv', '_new.csv')}"
        try:
            df.to_csv(alt_path, index=False)
            print(f"Results saved to alternative file: {alt_path}")
        except:
            print("Could not save to alternative file either.")
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Group statistics
    print("\nBy Group:")
    for group_name, group_results in [
        ("Discharge at Peak", group1_results),
        ("Charge at Cheap", group2_results),
        ("Between Peaks", group3_results),
        ("Idling", group4_results),
        ("Misaligned", group5_results),
        ("Extreme", group6_results),
    ]:
        if group_results:
            group_df = pd.DataFrame(group_results)
            print(f"\n{group_name}:")
            print(f"  Avg Reward: {group_df['reward'].mean():.2f}")
            print(f"  Avg Net Profit/MWh: {group_df['net_profit_per_mwh'].mean():.2f} EUR/MWh")
            print(f"  Profitable scenarios: {(group_df['net_profit_per_mwh'] > 0).sum()}/{len(group_df)}")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total scenarios: {len(df)}")
    print(f"Highest reward: {df.loc[df['reward'].idxmax(), 'scenario']} ({df['reward'].max():.2f})")
    print(f"Lowest reward: {df.loc[df['reward'].idxmin(), 'scenario']} ({df['reward'].min():.2f})")
    print(f"Highest net profit/MWh: {df.loc[df['net_profit_per_mwh'].idxmax(), 'scenario']} ({df['net_profit_per_mwh'].max():.2f} EUR/MWh)")
    print(f"Lowest net profit/MWh: {df.loc[df['net_profit_per_mwh'].idxmin(), 'scenario']} ({df['net_profit_per_mwh'].min():.2f} EUR/MWh)")
    print(f"Profitable scenarios: {(df['net_profit_per_mwh'] > 0).sum()}/{len(df)}")
    print(f"Unprofitable scenarios: {(df['net_profit_per_mwh'] < 0).sum()}/{len(df)}")
    
    # Unprofitable scenarios
    negative_profit = df[df['net_profit_per_mwh'] < 0]
    if len(negative_profit) > 0:
        print("\n" + "=" * 80)
        print("UNPROFITABLE SCENARIOS (degradation > arbitrage)")
        print("=" * 80)
        print(negative_profit[['scenario', 'net_profit_per_mwh', 'arbitrage_per_mwh', 'degradation_per_mwh', 'dod_current']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    df = run_all_scenarios()

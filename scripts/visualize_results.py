"""
Simple visualization script for BESS+PV operational results.
Focuses on operational and financial metrics for business presentations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style for clean, professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11


def plot_soc_and_prices(hourly_csv_path: str, output_path: Path, season_name: str = None, policy: str = "Energiabőrze"):
    """
    Plot SOC, day-ahead energy prices, and reserve prices over 24 hours.
    
    Args:
        hourly_csv_path: Path to hourly CSV file
        output_path: Where to save the plot
        season_name: Optional season name for title
        policy: Optional policy name to compare (default: "Energiabőrze"). 
                If provided, loads from eval_td3_energiaborze/hourly_data.csv and adds as second SOC line.
    """
    df = pd.read_csv(hourly_csv_path)
    
    # Load policy comparison data if requested
    df_policy = None
    policy_soc = None
    if policy:
        try:
            eval_csv = Path("results/eval_td3_energiaborze/hourly_data.csv")
            if eval_csv.exists():
                df_eval = pd.read_csv(eval_csv)
                
                # Find matching date - try different date column names
                date_col = None
                if 'test_day' in df.columns:
                    date_col = 'test_day'
                    main_date = df[date_col].iloc[0]
                elif 'eval_date' in df.columns:
                    date_col = 'eval_date'
                    main_date = df[date_col].iloc[0]
                elif 'date' in df.columns:
                    date_col = 'date'
                    main_date = df[date_col].iloc[0]
                else:
                    print(f"Warning: Could not find date column in main CSV. Skipping policy comparison.")
                    policy = None
                
                if policy and date_col:
                    # Match policy name (case-insensitive, handle variations)
                    policy_name = None
                    for p in df_eval['policy'].unique():
                        if policy.lower() in p.lower() or p.lower() in policy.lower():
                            policy_name = p
                            break
                    
                    if policy_name:
                        # Filter for matching date and policy
                        df_policy = df_eval[
                            (df_eval['test_day'] == main_date) & 
                            (df_eval['policy'] == policy_name)
                        ].copy()
                        
                        if len(df_policy) > 0:
                            df_policy = df_policy.sort_values('hour').reset_index(drop=True)
                            if 'soc_before' in df_policy.columns:
                                policy_soc = df_policy['soc_before'].values * 100
                            elif 'soc_raw' in df_policy.columns:
                                policy_soc = df_policy['soc_raw'].values * 100
                            else:
                                print(f"Warning: Could not find SOC column in policy data. Skipping policy comparison.")
                                policy = None
                        else:
                            print(f"Warning: No data found for policy '{policy_name}' on date {main_date}. Skipping policy comparison.")
                            policy = None
                    else:
                        print(f"Warning: Policy '{policy}' not found in evaluation results. Available policies: {df_eval['policy'].unique()}")
                        policy = None
            else:
                print(f"Warning: Evaluation results CSV not found at {eval_csv}. Skipping policy comparison.")
                policy = None
        except Exception as e:
            print(f"Warning: Error loading policy comparison data: {e}. Skipping policy comparison.")
            policy = None
    
    # Filter by season if specified
    if season_name and 'season_name' in df.columns:
        df = df[df['season_name'] == season_name].copy()
        title_suffix = f" - {season_name.capitalize()}"
    else:
        title_suffix = ""
    
    # Sort by hour to ensure correct order
    df = df.sort_values('hour').reset_index(drop=True)
    
    # Extract data
    hours = df['hour'].values
    # Use 'soc_before' (SOC at START of hour) to show the state BEFORE the action
    # This way: SOC at hour X shows the state going INTO hour X, and the action during hour X
    # will change it to the SOC at hour X+1. This aligns hours with actions correctly.
    # Alternative: use 'soc_raw' which should be the same as 'soc_before'
    if 'soc_before' in df.columns:
        soc = df['soc_before'].values * 100  # Convert to percentage
    else:
        soc = df['soc_raw'].values * 100  # Fallback to soc_raw if soc_before not available
    energy_prices = df['price_em_raw'].values  # EUR/MWh
    # Get PV production (use p_pv_raw if available, otherwise p_pv_grid)
    if 'p_pv_raw' in df.columns:
        pv_production = df['p_pv_raw'].values  # MW
    elif 'p_pv_grid' in df.columns:
        pv_production = df['p_pv_grid'].values  # MW
    else:
        raise ValueError("No PV production column found (p_pv_raw or p_pv_grid)")
    
    # Detect discharge periods (p_battery > threshold, e.g., > 0.1 MW to ignore noise)
    if 'p_battery' in df.columns:
        p_battery = df['p_battery'].values
        discharge_threshold = 0.1  # MW - ignore very small values (noise)
        is_discharging = p_battery > discharge_threshold
    else:
        # Fallback: detect discharge by SOC decreasing
        is_discharging = np.diff(np.concatenate([[soc[0]], soc])) < -0.5  # SOC drop > 0.5%
    
    # Create figure with 3 stacked subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Top subplot: SOC (0-100%)
    color_soc = '#2ecc71'  # Green
    ax1.set_ylabel('State of Charge (%)', fontsize=13, fontweight='bold', color=color_soc)
    ax1.plot(hours, soc, color=color_soc, linewidth=3, marker='o', 
             markersize=6, label='Battery SOC', zorder=3)
    
    # Add policy comparison line if available
    if policy and policy_soc is not None and len(policy_soc) == len(hours):
        color_policy = '#e74c3c'  # Red
        policy_label = policy  # Use policy name as-is (already correct for "Energiabőrze")
        ax1.plot(hours, policy_soc, color=color_policy, linewidth=3, marker='s', 
                 markersize=6, label=f'{policy_label} SOC', linestyle='--', zorder=3)
    
    ax1.tick_params(axis='y', labelcolor=color_soc)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='both')
    ax1.legend(loc='upper left', framealpha=0.95, fancybox=True, shadow=True)
    
    # Middle subplot: Day-Ahead Energy Prices (EUR/MWh)
    color_energy = '#3498db'  # Blue
    ax2.set_ylabel('Day-Ahead Energy Price (EUR/MWh)', fontsize=13, fontweight='bold', color=color_energy)
    ax2.plot(hours, energy_prices, color=color_energy, linewidth=2.5, 
             marker='s', markersize=5, label='Day-Ahead Energy Price', 
             linestyle='-', zorder=2)
    ax2.tick_params(axis='y', labelcolor=color_energy)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='both')
    ax2.legend(loc='upper left', framealpha=0.95, fancybox=True, shadow=True)
    
    # Bottom subplot: PV Production (MW)
    color_pv = '#f39c12'  # Orange/Amber for PV
    ax3.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax3.set_ylabel('PV Production (MW)', fontsize=13, fontweight='bold', color=color_pv)
    ax3.plot(hours, pv_production, color=color_pv, linewidth=2.5, 
             marker='^', markersize=5, label='PV Production', 
             linestyle='--', zorder=2)
    ax3.tick_params(axis='y', labelcolor=color_pv)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='both')
    ax3.legend(loc='upper left', framealpha=0.95, fancybox=True, shadow=True)
    
    # Set x-axis properties for all subplots (shared)
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24))  # Show all hours 0-23
        ax.set_xticklabels([str(h) for h in range(0, 24)])  # Explicit string labels
        ax.tick_params(axis='x', labelsize=10)
    
    # Add title to the figure
    eval_date = df['eval_date'].iloc[0] if 'eval_date' in df.columns else ""
    title = f"Battery State of Charge, Energy Prices, and PV Production Throughout the Day{title_suffix}"
    if eval_date:
        title += f"\nDate: {eval_date}"
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    
    # Add shading for discharge periods (detected from data) - apply to all subplots
    # Find consecutive discharge hours and shade them
    discharge_periods = []
    in_discharge = False
    start_hour = None
    
    for i, h in enumerate(hours):
        if is_discharging[i]:
            if not in_discharge:
                start_hour = h
                in_discharge = True
        else:
            if in_discharge:
                # End of discharge period - use previous hour as end
                if i > 0:
                    discharge_periods.append((start_hour, hours[i-1]))
                else:
                    discharge_periods.append((start_hour, start_hour))
                in_discharge = False
    
    # Handle case where discharge continues to end of day
    if in_discharge:
        discharge_periods.append((start_hour, hours[-1]))
    
    # Shade all discharge periods (draw behind lines with zorder=0) on all subplots
    for ax in [ax1, ax2, ax3]:
        for i, (start, end) in enumerate(discharge_periods):
            ax.axvspan(start - 0.5, end + 0.5, alpha=0.15, color='orange', zorder=0)
    
    # Add annotation for each discharge period (only on top subplot to avoid clutter)
    for start, end in discharge_periods:
        mid_hour = (start + end) / 2
        duration = end - start + 1
        if duration > 1:
            ax1.text(mid_hour, 95, f'Discharge\n({int(start)}-{int(end)})', ha='center', va='top', 
                    fontsize=9, alpha=0.7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax1.text(mid_hour, 95, 'Discharge', ha='center', va='top', 
                    fontsize=9, alpha=0.7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize BESS operational results")
    parser.add_argument("--hourly-csv", type=str, required=True,
                       help="Path to hourly CSV file")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output path for the plot")
    parser.add_argument("--season", type=str, default=None,
                       choices=['winter', 'spring', 'summer', 'fall'],
                       help="Filter by season (optional)")
    parser.add_argument("--policy", type=str, default="Energiabőrze",
                       help="Policy name to compare (default: 'Energiabőrze'). Set to empty string to disable.")
    args = parser.parse_args()
    
    # Convert empty string to None to disable policy comparison
    policy_arg = args.policy if args.policy else None
    
    plot_soc_and_prices(args.hourly_csv, args.output, args.season, policy_arg)


if __name__ == "__main__":
    main()


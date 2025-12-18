# Realistic PV generation based on solar physics: sunrise/sunset times, solar elevation angle
# Budapest, Hungary: latitude ~47.5°N, longitude ~19.0°E
# Uses realistic day length and solar elevation angle for bell-shaped curve

production_df_hourly = dayahead_df.copy()
rng_cloud = np.random.default_rng(12345)
unique_prod_dates = dayahead_df['datetime'].dt.date.unique()
cloud_factor_by_date = {d: 0.5 + 0.5 * rng_cloud.beta(5.0, 2.0) for d in unique_prod_dates}

# Budapest approximate sunrise/sunset times by month (local time, accounting for DST)
# Format: (sunrise_hour, sunset_hour) - approximate values
# Based on actual solar data for Budapest, Hungary
SUNRISE_SUNSET = {
    1: (7.5, 16.5),   # Winter: ~7:30-16:30 (8.5h daylight)
    2: (7.0, 17.5),   # Late winter: ~7:00-17:30 (10.5h)
    3: (6.0, 18.0),   # Spring: ~6:00-18:00 (12h)
    4: (5.5, 19.0),   # Spring: ~5:30-19:00 (13.5h)
    5: (5.0, 20.0),   # Late spring: ~5:00-20:00 (15h)
    6: (4.75, 20.75), # Summer: ~4:45-20:45 (16h)
    7: (5.0, 20.5),   # Summer: ~5:00-20:30 (15.5h)
    8: (5.5, 19.5),   # Late summer: ~5:30-19:30 (14h)
    9: (6.0, 19.0),   # Fall: ~6:00-19:00 (13h)
    10: (6.5, 18.0),  # Fall: ~6:30-18:00 (11.5h)
    11: (7.0, 16.5),  # Late fall: ~7:00-16:30 (9.5h)
    12: (7.5, 16.0),  # Winter: ~7:30-16:00 (8.5h)
}

def generate_pv_power_realistic(row):
    """
    Generate realistic PV power based on solar physics.
    Uses sunrise/sunset times and solar elevation angle for bell-shaped curve.
    
    Key improvements:
    - Zero production at night (strictly enforced)
    - Seasonal sunrise/sunset times (not hardcoded 6-18)
    - Bell-shaped curve based on solar elevation angle (not flat plateau)
    - Realistic day length variations
    """
    dt = row['datetime']
    month = dt.month
    hour = dt.hour + dt.minute / 60.0  # Decimal hour for precision
    weekday = dt.weekday()
    
    # Get sunrise and sunset for this month
    sunrise, sunset = SUNRISE_SUNSET[month]
    
    # Zero production outside daylight hours (strictly enforced - no exceptions)
    if hour < sunrise or hour >= sunset:
        return 0.0
    
    # Seasonal capacity (MW) - accounts for solar irradiance variations
    if month in [12, 1, 2]:
        max_capacity = 15.0  # Winter: lower irradiance, shorter days
    elif month in [3, 4, 5]:
        max_capacity = 13.5  # Spring: moderate irradiance (scaled from 18.0)
    elif month in [6, 7, 8]:
        max_capacity = 15.0  # Summer: peak irradiance, longest days (scaled from 20.0)
    else:
        max_capacity = 12.0  # Fall: moderate irradiance (scaled from 16.0)
    
    # Calculate solar elevation angle approximation
    # Solar noon is approximately at (sunrise + sunset) / 2
    solar_noon = (sunrise + sunset) / 2.0
    day_length = sunset - sunrise
    
    # Bell-shaped curve based on solar elevation angle
    # Uses cosine approximation: power ∝ cos(angle from noon)
    # This creates a realistic bell curve, not a flat plateau
    if day_length > 0:
        # Normalized time: -1 at sunrise/sunset, 0 at noon
        normalized_time = 2.0 * (hour - sunrise) / day_length - 1.0
        # Solar elevation approximation: bell curve using cosine
        # At noon: normalized_time = 0, elevation = max (cos(0) = 1)
        # At sunrise/sunset: normalized_time = ±1, elevation = 0 (cos(π/2) = 0)
        solar_factor = np.cos(normalized_time * np.pi / 2.0)
        # Ensure non-negative and smooth curve
        solar_factor = max(0.0, solar_factor)
        # Apply power law to make curve more realistic (steeper at edges)
        # This accounts for atmospheric effects and panel efficiency at low angles
        solar_factor = solar_factor ** 1.2
    else:
        solar_factor = 0.0
    
    # Weather (daily cloud clearness 0.5–1.0)
    # Represents daily weather conditions (cloudy vs clear day)
    clear = float(cloud_factor_by_date.get(dt.date(), 0.8))
    
    # Mild weekend factor (slight reduction due to potential curtailment)
    weekend_factor = 0.98 if weekday >= 5 else 1.0
    
    # Base power from solar physics
    base_power = max_capacity * solar_factor * clear * weekend_factor
    
    # Add realistic noise (clouds, atmospheric conditions, passing clouds)
    # Noise is proportional to expected power (more variation at higher power)
    if month in [12, 1, 2]:
        noise_std = 0.08 * base_power  # Higher relative noise in winter (more clouds)
    elif month in [6, 7, 8]:
        noise_std = 0.04 * base_power  # Lower relative noise in summer (clearer skies)
    else:
        noise_std = 0.06 * base_power
    
    # Add noise (but ensure it doesn't go negative)
    power = base_power + np.random.normal(0.0, noise_std)
    power = max(0.0, power)  # Ensure non-negative (can't have negative generation)
    
    # Final clip to capacity (accounting for inverter limits)
    return float(np.clip(power, 0.0, max_capacity))

# Recompute PV with realistic generator (reproducible noise)
np.random.seed(42)
production_df_hourly['pv_power_mw'] = production_df_hourly.apply(generate_pv_power_realistic, axis=1)

print(f"Realistic PV: range {production_df_hourly['pv_power_mw'].min():.2f}–{production_df_hourly['pv_power_mw'].max():.2f} MW")
print(f"Zero production hours (night): {(production_df_hourly['pv_power_mw'] == 0).sum()} / {len(production_df_hourly)}")
print(f"Percentage of night hours: {100 * (production_df_hourly['pv_power_mw'] == 0).sum() / len(production_df_hourly):.1f}%")
print(f"\nSample daily profile (first 24 hours):")
production_df_hourly[['datetime', 'pv_power_mw']].head(24)


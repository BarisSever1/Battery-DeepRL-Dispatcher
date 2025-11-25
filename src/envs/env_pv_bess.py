"""
Battery Energy Storage System (BESS) + Renewable Energy Source (RES) Environment

This environment simulates a co-located solar PV + battery system participating in:
- Day-Ahead Energy Market (arbitrage)
- Ancillary Services / Reserve Market
- RES generation sales

The agent learns to optimize daily operation by deciding how to allocate battery power
between energy trading and reserve provision while managing degradation and constraints.

Based on the TD3+LSTM approach for multi-market co-optimization.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces


class BESSEnv(gym.Env):
    """
    Gymnasium environment for BESS + RES co-optimization.
    
    State: 21-dimensional vector (normalized to [-1, 1])
        - Temporal: k, weekday, season
        - Prices: price_em, price_as
        - Operations: p_res_total, soc, dod
        - Daily context: morning/evening max prices, argmax, min price, argmin, reserve min/max
        - Planning aids: time_to_peak_hour, 6h price trend, delta to next morning/evening peaks,
          prior-step SOC (to expose delta SOC directly)
    
    Action: Continuous value δ ∈ [-1, 1]
        - δ < 0: Charge (magnitude scales charging power)
        - δ > 0: Discharge (magnitude scales discharging power)
        - δ = 0: Idle
    
    Reward: Total profit - degradation cost
        = Revenue_RES + Revenue_Energy + Revenue_Reserve - Cost_Degradation
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        data_path: str = 'data/processed/training_features_normalized_train.parquet',
        config_path: str = 'config/limits.yaml',
        norm_params_path: str = 'data/processed/normalization_params.yaml',
        seed: Optional[int] = None,
        degradation_model: str = "nonlinear",
    ):
        """
        Initialize the BESS environment.
        
        Args:
            data_path: Path to normalized feature dataset
            config_path: Path to system limits configuration
            norm_params_path: Path to normalization parameters
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Load data and configuration
        self._load_data(data_path)
        self._load_config(config_path)
        self._load_norm_params(norm_params_path)
        
        # Degradation model selection ("nonlinear" default, or "linear")
        self.degradation_model = degradation_model.lower().strip()
        # Target SOC for optional end-of-day shaping
        self.target_soc = 0.50  # default target
        self.terminal_soc_penalty_eur = 3000.0  # € penalty per unit SOC deviation at day end
        self.cycle_penalty_eur_per_cycle = 2500.0  # € penalty per extra full cycle beyond target
        self.prepeak_morning_target_soc = 0.75
        self.prepeak_evening_target_soc = 0.80
        self.target_daily_throughput_mwh = 2.0 * self.E_capacity
        self.throughput_penalty_eur_per_mwh = 150.0

        # Define action and observation spaces
        # Action: δ ∈ [-1, 1] - negative = charge, positive = discharge, 0 = idle
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation: 21-dimensional state vector (normalized)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(21,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_day = None
        self.current_hour = 0
        self.max_hours = 24
        
        # Battery state
        self.soc = 0.5  # State of charge (0-1)
        self.dod_morning = 0.0  # DOD for morning discharge
        self.dod_evening = 0.0  # DOD for evening discharge
        
        # Episode metrics
        self.episode_revenue = 0.0
        self.episode_degradation = 0.0
        self.episode_info = []
        
        # Daily throughput tracking for cycle budget penalty
        self.daily_throughput = 0.0
        
    def _load_data(self, data_path: str):
        """Load normalized feature dataset."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.data = pd.read_parquet(path)
        
        # Group by date to get daily episodes
        self.data['date'] = self.data['datetime'].dt.date
        self.dates = self.data['date'].unique()
        
        print(f"Loaded {len(self.data)} hourly records")
        print(f"Available dates: {len(self.dates)} days")
        
    def _load_config(self, config_path: str):
        """Load system configuration and limits."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Battery parameters
        bess = config['bess']
        self.P_bess_max = bess['power_rating_mw']  # 20 MW
        self.E_capacity = bess['energy_capacity_mwh']  # 60 MWh
        self.soc_min = bess['soc_min']  # 0.10
        self.soc_max = bess['soc_max']  # 0.90
        self.eta_charge = bess.get('charge_efficiency', 0.97)  # 0.97
        self.eta_discharge = bess.get('discharge_efficiency', 0.97)  # 0.97
        self.eta_rt = bess.get('round_trip_efficiency', self.eta_charge * self.eta_discharge)  # ~0.94
        
        # RES parameters
        res = config['res']
        self.P_res_max = res.get('pv_ac_capacity_mw', res.get('pv_capacity_mw', 20.0))  # 20 MW
        
        # Grid/POI limits
        grid = config['grid']
        self.P_poi_max = grid['poi_power_limit_mw']  # 20 MW
        self.allow_grid_import = grid['allow_grid_import']  # True
        self.P_import_max = grid.get('max_grid_import_mw', 20.0)  # 20 MW
        
        # Converter/inverter nominal ratings and efficiency aliases (paper naming)
        self.P_conv_nom = self.P_bess_max
        self.P_inv_nom = grid.get('total_inverter_rating_mw', self.P_poi_max)
        self.eta_ch = self.eta_charge
        self.eta_dis = self.eta_discharge
        self.prepeak_window_hours = 2
        self.post_evening_time_placeholder = 21.0
        # Time step (fixed for hourly operation)
        self.dt = 1.0  # 1 hour time step
        
        # Degradation cost parameters (read from limits.yaml → degradation: {...})
        deg = config.get('degradation', {}) if isinstance(config, dict) else {}
        # RL stabilizers patch — degradation calibration
        try:
            self.capex_eur_per_mwh = float(deg.get('capex_eur_per_mwh', 175000.0))
        except Exception:
            self.capex_eur_per_mwh = 175000.0
        try:
            self.dod_max_for_cost = float(deg.get('dod_max_for_cost', 0.80))
        except Exception:
            self.dod_max_for_cost = 0.80
        try:
            self.n_cycles_at_dod_max = int(deg.get('n_cycles_at_dod_max', 5000))
        except Exception:
            self.n_cycles_at_dod_max = 5000

        # Average DOD and cycles for linear degradation calibration
        try:
            self.dod_avg_for_linear = float(deg.get('dod_avg_for_linear', 0.60))
        except Exception:
            self.dod_avg_for_linear = 0.60
        try:
            self.n_cycles_avg_for_linear = int(deg.get('n_cycles_avg_for_linear', 6000))
        except Exception:
            self.n_cycles_avg_for_linear = 6000

        print(f"Loaded config: {self.P_bess_max} MW / {self.E_capacity} MWh BESS")
        
    def _load_norm_params(self, norm_params_path: str):
        """Load normalization parameters for denormalization."""
        path = Path(norm_params_path)
        if not path.exists():
            raise FileNotFoundError(f"Normalization params not found: {norm_params_path}")
        
        with open(path, 'r') as f:
            self.norm_params = yaml.safe_load(f)
        
        loaded_keys = set(self.norm_params.get('features', {}).keys())
        print(f"Loaded normalization parameters for {len(loaded_keys)} features")

        # Validate that all dataset-driven observation features have stats.
        # Excludes env-derived: 'soc', 'dod', and time_to_peak_hour.
        expected_norm_features = {
            'k', 'weekday', 'season',
            'price_em', 'price_as', 'p_res_total',
            'price_em_max_morning', 'price_em_max_evening',
            'k_em_max_morning', 'k_em_max_evening',
            'price_em_min', 'k_em_min',
            'price_as_min', 'price_as_max',
            'time_to_peak_hour',
        }
        missing = sorted(list(expected_norm_features - loaded_keys))
        extra = sorted(list(loaded_keys - expected_norm_features))
        if missing:
            print(f"Warning: normalization params missing for features: {missing}")
        if extra:
            # Extra keys are allowed; just informative
            print(f"Note: normalization params include extra features: {extra}")
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode (new day).
        
        Args:
            seed: Random seed
            options: Additional options (e.g., specific date to use)
        
        Returns:
            observation: Initial state vector (16 dims)
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Select a random day or specific day from options
        if options and 'date' in options:
            self.current_day = options['date']
        else:
            self.current_day = np.random.choice(self.dates)
        
        # Get day's data
        self.day_data = self.data[self.data['date'] == self.current_day].reset_index(drop=True)
        # Cache denormalized day-ahead prices for reward shaping or diagnostics
        try:
            prices_norm = self.day_data['price_em'].astype(float).to_numpy()
            prices_raw = np.array([self._denormalize('price_em', float(v)) for v in prices_norm], dtype=float)
            self._day_prices_raw = prices_raw
            self.day_price_avg_raw = float(np.nanmean(prices_raw))
            self.day_price_low_q20_raw = float(np.nanquantile(prices_raw, 0.20))
            self.day_price_max_raw = float(np.nanmax(prices_raw))
            self.day_price_min_raw = float(np.nanmin(prices_raw))
            self.day_price_range_raw = float(self.day_price_max_raw - self.day_price_min_raw)
        except Exception:
            self._day_prices_raw = None
            self.day_price_avg_raw = float('nan')
            self.day_price_low_q20_raw = float('nan')
            self.day_price_max_raw = float('nan')
            self.day_price_min_raw = float('nan')
            self.day_price_range_raw = float('nan')

        # Compute peak discharge slots directly from today's price curve (fallback to dataset hours)
        try:
            prices = np.asarray(self._day_prices_raw, dtype=float)
            assert prices.shape[0] == 24
            self.k_morn = int(np.argmax(prices[:12]))                  # 0–11
            self.k_even = int(12 + np.argmax(prices[12:]))             # 12–23
        except Exception:
            try:
                k_m = int(round(self._denormalize('k_em_max_morning', self.day_data.loc[0, 'k_em_max_morning'])))
                k_e = int(round(self._denormalize('k_em_max_evening', self.day_data.loc[0, 'k_em_max_evening'])))
                self.k_morn = int(np.clip(k_m, 0, 23))
                self.k_even = int(np.clip(k_e, 0, 23))
            except Exception:
                self.k_morn = -1
                self.k_even = -1

        # Recompute cheapest hours for the selected day (for optional shaping/diagnostics)
        try:
            if self._day_prices_raw is not None and len(self._day_prices_raw) >= 2:
                prices_arr = np.asarray(self._day_prices_raw, dtype=float)
                cheap_threshold = float(np.nanquantile(prices_arr, 0.30))
                self.cheapest_hours = {int(idx) for idx, p in enumerate(prices_arr) if p <= cheap_threshold}
                if len(self.cheapest_hours) < 4:
                    price_sorted_asc = np.argsort(prices_arr)
                    self.cheapest_hours = set(int(h) for h in price_sorted_asc[:4])
            else:
                self.cheapest_hours = set()
        except Exception:
            self.cheapest_hours = set()
        if len(self.cheapest_hours) == 0:
            self.cheapest_hours = {0, 1, 2, 3}
        self._day_pv_raw = None

        # Build peak discharge windows (±1 hour around detected maxima)
        def _build_peak_window(center: int, start_min: int, end_max: int) -> set[int]:
            if center < 0:
                return set()
            start = max(start_min, center - 1)
            end = min(end_max, center + 1)
            return set(range(start, end + 1))

        self.morning_peak_hours = _build_peak_window(getattr(self, "k_morn", -1), 0, 11)
        self.evening_peak_hours = _build_peak_window(getattr(self, "k_even", -1), 12, 23)
        self.peak_hours = self.morning_peak_hours.union(self.evening_peak_hours)
        
        # Pre-peak hours (1-2 hours before each peak) - computed once
        k_morn = getattr(self, "k_morn", -1)
        k_even = getattr(self, "k_even", -1)
        self.pre_morning_hours = {max(0, k_morn - 2), max(0, k_morn - 1)} if k_morn > 0 else set()
        self.pre_evening_hours = {max(12, k_even - 2), max(12, k_even - 1)} if k_even > 12 else set()
        
        if len(self.day_data) != 24:
            print(f"Warning: Day {self.current_day} has {len(self.day_data)} hours instead of 24")
        
        # Reset episode state
        self.current_hour = 0
        # Start every day at mid-SOC to encourage symmetric arbitrage
        self.soc = 0.5
        self.soc_previous = 0.5
        # Reset daily DOD trackers
        self.dod_morning = 0.0
        self.dod_evening = 0.0
        # Initialize per-period SOC extrema for DOD tracking (Eq. 18: max-min over period)
        self.soc_min_morning = self.soc
        self.soc_max_morning = self.soc
        # Evening extrema are initialized upon first entry to evening period (hour >= 12)
        self.soc_min_evening = None
        self.soc_max_evening = None
        # If a daily reset helper exists, call it
        if hasattr(self, "_reset_daily_dod"):
            try:
                self._reset_daily_dod()  # optional hook
            except Exception:
                pass
        
        # Reset metrics
        self.episode_revenue = 0.0
        self.episode_degradation = 0.0
        self.episode_info = []
        
        # Reset daily throughput tracker
        self.daily_throughput = 0.0
        self.mode = "shaping_only"
        
        # Get initial observation
        obs = self._get_observation()
        info = {'day': str(self.current_day), 'hour': 0, 'soc': self.soc, 'mode': self.mode}
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation (21-dimensional vector).
        
        Returns:
            Normalized state vector matching features.yaml order plus env-derived signals.
        """
        if self.current_hour >= len(self.day_data):
            # Episode done, return last valid state
            self.current_hour = len(self.day_data) - 1
        
        hour_data = self.day_data.iloc[self.current_hour]

        # === Dynamic time-to-next-peak computation (match training normalization) ===
        try:
            k1_dyn, k2_dyn = getattr(self, "k_morn", None), getattr(self, "k_even", None)
            cur_h = int(self.current_hour)
            if k1_dyn is None or k2_dyn is None or cur_h < 0 or cur_h > 23:
                time_to_next_peak_hour = 0.0
            else:
                peak_hours = getattr(self, "peak_hours", set())
                if cur_h in peak_hours:
                    # Currently within peak window → treat as immediate peak
                    time_to_next_peak_hour = 0.0
                elif cur_h < k1_dyn:
                    time_to_next_peak_hour = float(max(0, k1_dyn - cur_h))
                elif cur_h < k2_dyn:
                    time_to_next_peak_hour = float(max(0, k2_dyn - cur_h))
                else:
                    # Past evening peak → hold a fixed high placeholder to signal next-day wait
                    time_to_next_peak_hour = float(self.post_evening_time_placeholder)
                time_to_next_peak_hour = float(np.clip(time_to_next_peak_hour, 0.0, 24.0))
        except Exception:
            time_to_next_peak_hour = 0.0
        # Normalize to [-1, 1] using stored min/max (default 0..21)
        try:
            t2p_params = self.norm_params['features']['time_to_peak_hour']
            t2p_min = float(t2p_params.get('min', 0.0))
            t2p_max = float(t2p_params.get('max', 21.0))
        except Exception:
            t2p_min, t2p_max = 0.0, 21.0
        t2p_range = max(t2p_max - t2p_min, 1e-6)
        t2p_norm = 2.0 * ((time_to_next_peak_hour - t2p_min) / t2p_range) - 1.0
        t2p_norm = float(np.clip(t2p_norm, -1.0, 1.0))

        # --- Future-aware price features ---
        price_range_scale = max(1.0, getattr(self, "day_price_range_raw", 1.0))
        trend_norm = 0.0
        delta_morning_norm = 0.0
        delta_evening_norm = 0.0
        price_trend = 0.0
        delta_morning = 0.0
        delta_evening = 0.0
        price_em_raw = None
        try:
            if self._day_prices_raw is not None and len(self._day_prices_raw) > self.current_hour:
                price_em_raw = float(self._day_prices_raw[self.current_hour])
            else:
                price_em_raw = float(self._denormalize('price_em', hour_data['price_em']))
        except Exception:
            price_em_raw = 0.0

        if price_em_raw is None:
            price_em_raw = 0.0

        try:
            if self._day_prices_raw is not None:
                future_window = self._day_prices_raw[self.current_hour + 1 : min(len(self._day_prices_raw), self.current_hour + 1 + 6)]
                if len(future_window) > 0:
                    future_avg = float(np.mean(future_window))
                    price_trend = future_avg - price_em_raw
                    trend_norm = float(np.clip(price_trend / price_range_scale, -1.0, 1.0))

                if 0 <= getattr(self, "k_morn", -1) < len(self._day_prices_raw):
                    if self.current_hour <= self.k_morn:
                        delta_morning = float(self._day_prices_raw[self.k_morn] - price_em_raw)
                        delta_morning_norm = float(np.clip(delta_morning / price_range_scale, -1.0, 1.0))
                if 0 <= getattr(self, "k_even", -1) < len(self._day_prices_raw):
                    if self.current_hour <= self.k_even:
                        delta_evening = float(self._day_prices_raw[self.k_even] - price_em_raw)
                        delta_evening_norm = float(np.clip(delta_evening / price_range_scale, -1.0, 1.0))
        except Exception:
            price_trend = 0.0
            delta_morning = 0.0
            delta_evening = 0.0
            trend_norm = 0.0
            delta_morning_norm = 0.0
            delta_evening_norm = 0.0

        # Get normalized features from data (already normalized)
        obs = np.array([
            hour_data['k'],                    # 1. Hour of day
            hour_data['weekday'],              # 2. Day of week
            hour_data['season'],               # 3. Season
            hour_data['price_em'],             # 4. Day-ahead price
            hour_data['price_as'],             # 5. Reserve price
            hour_data['p_res_total'],          # 6. RES generation
            self._normalize_soc(self.soc),     # 7. Battery SOC (environment state)
            self._normalize_dod(self._get_current_dod()),  # 8. DOD (environment state)
            hour_data['price_em_max_morning'], # 9. Morning max price
            hour_data['price_em_max_evening'], # 10. Evening max price
            hour_data['k_em_max_morning'],     # 11. Morning max hour
            hour_data['k_em_max_evening'],     # 12. Evening max hour
            hour_data['price_em_min'],         # 13. Daily min price
            hour_data['k_em_min'],             # 14. Daily min hour
            hour_data['price_as_min'],         # 15. Daily reserve min
            hour_data['price_as_max'],         # 16. Daily reserve max
            t2p_norm,                          # 17. Normalized time to next peak
            trend_norm,                        # 18. Future price trend (6h avg ratio)
            delta_morning_norm,                # 19. Remaining spread to morning peak
            delta_evening_norm,                # 20. Remaining spread to evening peak
            self._normalize_soc(self.soc_previous),  # 21. SOC before action (gives delta context)
        ], dtype=np.float32)

        self._latest_obs_future_features = {
            "time_to_peak_hour_raw": float(time_to_next_peak_hour),
            "future_price_trend_raw": float(price_trend),
            "delta_to_morning_peak_raw": float(delta_morning),
            "delta_to_evening_peak_raw": float(delta_evening),
        }
        
        return obs
    
    def _normalize_soc(self, soc: float) -> float:
        """Normalize SOC to [-1, 1] range."""
        # SOC is already in [0, 1], map to [-1, 1]
        return 2.0 * soc - 1.0
    
    def _normalize_dod(self, dod: float) -> float:
        """Normalize DOD to [-1, 1] range."""
        # DOD is in [0, 1], map to [-1, 1]
        return 2.0 * dod - 1.0
    
    def _get_current_dod(self) -> float:
        """Get current DOD based on time of day."""
        k = self.current_hour
        if k < 12:
            return self.dod_morning
        else:
            return self.dod_evening

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with ONLY day-ahead energy market + reserve availability.
        PV → grid revenue uses price_em (no PPA).
        δ ∈ [-1, 1]: negative = charge, positive = discharge, 0 = idle.
        Reserve is derived from remaining headroom (β=1).
        """
        # === ACTION PARSING: RL-friendly constant scaling ===
        # delta ∈ [-1,1] maps linearly to battery power [-Pmax, +Pmax]
        delta = float(np.clip(action[0], -1.0, 1.0))
        p_battery_cmd = delta * self.P_bess_max  # MW (unconstrained command)

        # --- 1) Fetch current data (denormalized)
        hour = self.current_hour
        row = self.day_data.iloc[hour]

        price_em = self._denormalize('price_em', row['price_em'])      # €/MWh (energy market)
        price_as = self._denormalize('price_as', row['price_as'])      # €/MW·h (reserve)
        p_pv_t   = self._denormalize('p_res_total', row['p_res_total'])  # MW raw PV

        dt = self.dt
        Emax = self.E_capacity

        # === 2) APPLY PHYSICAL LIMITS ===
        # Max discharge available limited by SOC:
        pmax_dis = min(
            self.P_bess_max,
            self.P_conv_nom,
            max(0.0, (self.soc - self.soc_min) * Emax / dt),
        )
        # Max charge available limited by SOC:
        pmax_ch = min(
            self.P_bess_max,
            self.P_conv_nom,
            max(0.0, (self.soc_max - self.soc) * Emax / dt),
        )
        if p_battery_cmd >= 0:
            p_battery = min(p_battery_cmd, pmax_dis)
        else:
            p_battery = max(p_battery_cmd, -pmax_ch)

        # --- 3) AC-coupled routing with PV priority
        if p_battery >= 0.0:
            # Discharging: all PV goes to grid; battery exports to EM
            p_pv_grid = p_pv_t
            p_bess_em = p_battery
        else:
            # Charging: PV first, then grid import
            pv_to_charge = min(p_pv_t, -p_battery)
            p_pv_grid = p_pv_t - pv_to_charge
            p_bess_em = -( (-p_battery) - pv_to_charge )  # negative → buy from EM

        # === Reserve request derived from remaining headroom (paper-aligned) ===
        # Request full capacity; constraints will reduce to feasible remainder
        p_reserve_req = self.P_bess_max

        # --- 5) Apply constraints (pass PV-to-grid for POI check)
        p_battery_feas, p_reserve_feas = self._apply_constraints(
            p_battery=p_battery,
            p_reserve=p_reserve_req,
            p_pv_grid=p_pv_grid
        )

        # If exchange was clipped, re-route AC flows consistently
        if p_battery_feas != p_battery:
            p_battery = p_battery_feas
            if p_battery >= 0.0:
                p_pv_grid = p_pv_t
                p_bess_em = p_battery
            else:
                pv_to_charge = min(p_pv_t, -p_battery)
                p_pv_grid = p_pv_t - pv_to_charge
                p_bess_em = -( (-p_battery) - pv_to_charge )

            poi_headroom = max(0.0, self.P_poi_max - (p_pv_grid + max(p_bess_em, 0.0)))
            p_reserve_feas = min(p_reserve_feas, poi_headroom)

        p_reserve = p_reserve_feas

        # --- 6) Update SOC & DOD
        # DOD fix: now uses self.soc_previous and self.soc; removed unused parameters.
        self.soc_previous = self.soc
        self.soc = self._update_soc(p_battery)
        self._reset_dod_daily_if_needed(hour)
        self._update_dod()
        delta_soc = abs(self.soc - self.soc_previous)
        energy_mwh = abs(p_battery) * dt
        soc_low = self.soc_min + 0.05
        soc_high = self.soc_max - 0.05

        # --- 7) Economics (NO PPA): all PV-to-grid at price_em
        revenue_pv_grid  = p_pv_grid * price_em * dt          # €/step
        revenue_energy   = p_bess_em * price_em * dt          # €/step (neg if buying)
        revenue_reserve  = p_reserve  * price_as * dt         # €/step
        cost_degradation = self._calculate_degradation_cost(p_battery)
        
        # raw_reward = (
        #     revenue_pv_grid
        #     + revenue_energy
        #     + revenue_reserve
        #     - cost_degradation
        # )
        self.daily_throughput += abs(p_battery) * dt
        
        # --- Price structure factors ---
        price_min = getattr(self, "day_price_min_raw", price_em)
        price_max = getattr(self, "day_price_max_raw", price_em)
        price_range = max(1e-6, price_max - price_min)
        cheap_factor = float(np.clip((price_max - price_em) / price_range, 0.0, 1.0))
        expensive_factor = float(np.clip((price_em - price_min) / price_range, 0.0, 1.0))

        # --- Upcoming peak hour ---
        next_peak_hour = None
        if hasattr(self, "peak_hours"):
            for h in sorted(self.peak_hours):
                if h >= hour:
                    next_peak_hour = h
                    break
        hours_to_peak = None
        if next_peak_hour is not None:
            hours_to_peak = int(max(0, next_peak_hour - hour))
        peak_hours = getattr(self, "peak_hours", set())

        # --- Reward Shaping: td3_tuesday4 version (price-driven arbitrage) ---
        reward = 0.0

        # 1) Core arbitrage: charge when cheap, discharge when expensive
        if p_battery < 0:  # Charging
            reward += cheap_factor * energy_mwh * 20.0

        if p_battery > 0:  # Discharging
            reward += expensive_factor * energy_mwh * 20.0

            # Extra bonus for discharging during identified peak hours
            if hour in peak_hours:
                reward += energy_mwh * 30.0  # Strong peak discharge bonus

        # 2) Pre-peak accumulation incentive (aim for high SOC before peak)
        if hours_to_peak is not None and hours_to_peak in (1, 2):
            soc_target = 0.85
            soc_gap = max(0.0, soc_target - self.soc)
            if soc_gap > 0.0:
                if p_battery < 0:  # Charging toward target
                    reward += soc_gap * energy_mwh * 10.0
                elif p_battery > 0 and soc_gap > 0.1:  # Discharging when should hold
                    reward -= 20.0 * energy_mwh

        # 3) Cheap hour opportunity cost: penalize not charging enough when cheap
        cheap_hours = getattr(self, "cheapest_hours", {0, 1, 2, 3})
        if hour in cheap_hours and self.soc < 0.80 and p_battery >= 0:
            # If we're in a cheap window, not charging toward a comfortable SOC is bad
            reward -= (0.80 - self.soc) * 3.0

        # 4) SOC boundary penalties (soft safety band 20%–80%)
        soc_low_bound = 0.20
        soc_high_bound = 0.80
        if self.soc < soc_low_bound:
            reward -= (soc_low_bound - self.soc) * 30.0
        if self.soc > soc_high_bound:
            reward -= (self.soc - soc_high_bound) * 40.0

        # 5) Terminal SOC target: end-of-day around 50%
        if (hour + 1) >= self.max_hours:
            target_soc = 0.50
            reward -= 40.0 * abs(self.soc - target_soc)

        # 6) Linear scaling (td3_tuesday4): keep reward magnitudes moderate
        reward = reward / 25.0

        future_features = getattr(self, "_latest_obs_future_features", {}) or {}
        future_price_trend = float(future_features.get("future_price_trend_raw", 0.0))
        delta_to_morning_peak = float(future_features.get("delta_to_morning_peak_raw", 0.0))
        delta_to_evening_peak = float(future_features.get("delta_to_evening_peak_raw", 0.0))
        time_to_peak_hour_raw = float(future_features.get("time_to_peak_hour_raw", 0.0))



        # --- 8) Log & advance
        self.episode_revenue += (revenue_pv_grid + revenue_energy + revenue_reserve)
        self.episode_degradation += cost_degradation

        info = {
            'hour': hour,
            'delta': delta,
            'soc_previous': self.soc_previous,
            'soc_before': self.soc_previous,
            'soc': self.soc,
            'p_pv_raw': p_pv_t,
            'p_pv_grid': p_pv_grid,
            'p_battery': p_battery,
            'p_bess_em': p_bess_em,
            'p_reserve': p_reserve,
            'price_em': price_em,
            'price_as': price_as,
            'price_em_raw': price_em,
            'revenue_pv_grid': revenue_pv_grid,
            'revenue_energy': revenue_energy,
            'revenue_reserve': revenue_reserve,
            'cost_degradation': cost_degradation,
            'cheap_factor': cheap_factor,
            'expensive_factor': expensive_factor,
            'delta_soc': delta_soc,
            'energy_mwh': energy_mwh,
            'peak_hour': hour in peak_hours if isinstance(peak_hours, set) else False,
            'hours_to_peak': hours_to_peak,
            'future_price_trend': future_price_trend,
            'delta_to_morning_peak': delta_to_morning_peak,
            'delta_to_evening_peak': delta_to_evening_peak,
            'time_to_peak_hour_raw': time_to_peak_hour_raw,
            'reward': reward,
            'reward_shaping': reward,
            'reward_base': 0.0,
            'reward_final': reward,
            # 'terminal_soc_penalty': terminal_soc_penalty,
            'is_cheap_hour': hour in self.cheapest_hours if hasattr(self, 'cheapest_hours') else False,
            'dod_morning': self.dod_morning,
            'dod_evening': self.dod_evening,
            'mode': getattr(self, "mode", "shaping_only"),
        }

        self.current_hour += 1
        terminated = self.current_hour >= self.max_hours
        truncated = False

        obs = self._get_observation()
        if terminated:   
            info['episode_revenue'] = self.episode_revenue
            info['episode_degradation'] = self.episode_degradation
            info['episode_profit'] = self.episode_revenue - self.episode_degradation

        return obs, float(reward), bool(terminated), bool(truncated), info

    
    def _denormalize(self, feature: str, normalized_value: float) -> float:
        """
        Denormalize a feature value from [-1, 1] to original range.
        
        Args:
            feature: Feature name
            normalized_value: Normalized value in [-1, 1]
        
        Returns:
            Original value
        """
        params = self.norm_params['features'][feature]
        min_val = params['min']
        max_val = params['max']
        
        # Inverse of: normalized = 2 * (value - min) / (max - min) - 1
        # value = ((normalized + 1) / 2) * (max - min) + min
        value = ((normalized_value + 1.0) / 2.0) * (max_val - min_val) + min_val
        
        return value
        
    def _apply_constraints(self, p_battery: float, p_reserve: float, p_pv_grid: float):
        """
        Apply battery, converter, SOC, and POI constraints (paper-aligned).
        Returns (p_battery_feasible, p_reserve_feasible)
        """
        # 1. SOC bounds and converter/battery limits
        pmax_dis = min(self.P_bess_max, self.P_conv_nom,
                       (self.soc - self.soc_min) * self.E_capacity / self.dt)
        pmax_ch  = min(self.P_bess_max, self.P_conv_nom,
                       (self.soc_max - self.soc) * self.E_capacity / self.dt)
        if p_battery > 0:
            p_battery = min(p_battery, pmax_dis)
        else:
            p_battery = max(p_battery, -pmax_ch)

        # 2. POI export constraint (Eq. 15)
        total_export = p_pv_grid + max(p_battery, 0.0) + p_reserve
        if total_export > self.P_poi_max:
            overflow = total_export - self.P_poi_max
            if p_reserve > 0:
                p_reserve = max(0.0, p_reserve - overflow)
                total_export = p_pv_grid + max(p_battery, 0.0) + p_reserve
            if total_export > self.P_poi_max:
                p_battery = max(0.0, self.P_poi_max - p_pv_grid - p_reserve)

        # 3. Reserve feasibility (Eqs. 25–26)
        p_soc_headroom = max(0.0, (self.soc - self.soc_min) * self.E_capacity / self.dt)
        p_dis_max_prime = min(self.P_bess_max, self.P_conv_nom, p_soc_headroom)
        poi_headroom = max(0.0, self.P_poi_max - (p_pv_grid + max(p_battery, 0.0)))
        # Converter upward headroom from current operating point:
        # If charging at -c, you must first cancel c before providing upward reserve.
        conv_up_headroom = max(0.0, self.P_conv_nom - max(0.0, -p_battery))
        p_reserve = min(p_reserve, p_dis_max_prime - max(p_battery, 0.0), poi_headroom, conv_up_headroom)

        return p_battery, p_reserve

    
    
    def _update_soc(self, p_battery: float) -> float:
        """
        Update the state of charge (SOC) based on current battery power.
        Positive p_battery = discharge to grid
        Negative p_battery = charge from grid/PV

        Returns:
            new_soc (float): Updated SOC value (fraction 0–1)
        """
        # Convert MW × h → MWh for this timestep
        dt_energy = p_battery * self.dt  # MWh (signed)

        if p_battery > 0:  # Discharging
            # Remove more energy from cells than delivered to grid
            soc_delta = -abs(dt_energy) / (self.E_capacity * self.eta_dis)
        elif p_battery < 0:  # Charging
            # Store less energy than drawn (charging losses)
            soc_delta = -dt_energy * self.eta_ch / self.E_capacity
        else:
            soc_delta = 0.0

        new_soc = self.soc + soc_delta
        # Clip within limits from limits.yaml
        new_soc = float(np.clip(new_soc, self.soc_min, self.soc_max))
        self.soc = new_soc
        return new_soc

    def _reset_dod_daily_if_needed(self, hour: int) -> None:
        """Reset daily DOD counters at the start of the day."""
        if hour == 0:
            self.dod_morning = 0.0
            self.dod_evening = 0.0
            # Reset extrema to the initial SOC at the start of the day
            self.soc_min_morning = self.soc
            self.soc_max_morning = self.soc
            self.soc_min_evening = None
            self.soc_max_evening = None

    def _update_dod(self):
        """
        Update depth-of-discharge per paper Eq. (18):
        DOD(part_i) = SOC_max(part_i) - SOC_min(part_i)

        We track running min/max SOC within the morning (hours 0-11) and evening
        (hours 12-23) periods, and set DOD as max-min for the respective period.
        """
        # DOD fix: now uses self.soc_previous and self.soc; removed unused parameters.
        soc_before = getattr(self, "soc_previous", self.soc)
        soc_after = self.soc

        if self.current_hour < 12:
            # Morning period
            self.soc_min_morning = min(self.soc_min_morning, soc_before, soc_after)
            self.soc_max_morning = max(self.soc_max_morning, soc_before, soc_after)
            self.dod_morning = float(self.soc_max_morning - self.soc_min_morning)
        else:
            # Initialize evening extrema upon first entry to evening period
            if self.soc_min_evening is None or self.soc_max_evening is None:
                # Initialize with the SOC at transition into evening
                self.soc_min_evening = min(soc_before, soc_after)
                self.soc_max_evening = max(soc_before, soc_after)
            else:
                self.soc_min_evening = min(self.soc_min_evening, soc_before, soc_after)
                self.soc_max_evening = max(self.soc_max_evening, soc_before, soc_after)
            self.dod_evening = float(self.soc_max_evening - self.soc_min_evening)

    def _z_factor_from_dod(self, dod_frac: float) -> float:
        """DOD bucket multiplier (piecewise)."""
        dod_pct = 100.0 * max(0.0, min(1.0, dod_frac))
        if dod_pct > 70:
            return 1.00
        elif 55 < dod_pct <= 70:
            return 0.75
        elif 40 < dod_pct <= 55:
            return 0.50
        elif 30 < dod_pct <= 40:
            return 0.40
        elif 10 < dod_pct <= 30:
            return 0.10
        else:  # 0–10%
            return 0.02

    def _calculate_degradation_cost(self, p_battery: float) -> float:
        """
        Wrapper to compute degradation cost based on selected model.
        """
        if self.degradation_model == "linear":
            return self._calculate_degradation_cost_linear(p_battery)
        return self._calculate_degradation_cost_nonlinear(p_battery)

    def _calculate_degradation_cost_linear(self, p_battery: float) -> float:
        """
        Linear per-throughput degradation cost (EUR/MWh), constant across DOD.
        Uses the same capex/cycle assumptions as the nonlinear path but without
        the DOD bucket multiplier. This makes TD3(Linear) comparable against
        TD3(Nonlinear) to isolate the impact of the degradation model itself.
        """
        Emax_MWh = float(self.E_capacity)
        CBESS = float(self.capex_eur_per_mwh) * Emax_MWh
        # Use average DOD and cycles for a stable linear cost independent of current DOD
        DOD_avg = float(getattr(self, 'dod_avg_for_linear', self.dod_max_for_cost))
        N_cycles_avg = float(getattr(self, 'n_cycles_avg_for_linear', self.n_cycles_at_dod_max))
        C_cyc_per_MWh = CBESS / max(1e-6, Emax_MWh * DOD_avg * N_cycles_avg)
        energy_throughput_MWh = abs(p_battery) * self.dt
        return float(energy_throughput_MWh * C_cyc_per_MWh)

    def _calculate_degradation_cost_nonlinear(self, p_battery: float) -> float:
        """
        Non-linear DOD-based degradation (paper Eqs. 16–19).
        Returns cost in EUR per step.
        """
        Emax_MWh = float(self.E_capacity)
        CBESS = float(self.capex_eur_per_mwh) * Emax_MWh
        DOD_max = float(self.dod_max_for_cost)
        N_cycles = float(self.n_cycles_at_dod_max)

        # Base cycling cost C_cyc (per MWh throughput at DOD_max)
        C_cyc_per_MWh = CBESS / max(1e-6, Emax_MWh * DOD_max * N_cycles)

        # Current DOD bucket and multiplier
        current_dod = self._get_current_dod()
        z = self._z_factor_from_dod(current_dod)
        Cdyn_per_MWh = z * C_cyc_per_MWh

        energy_throughput_MWh = abs(p_battery) * self.dt
        return float(energy_throughput_MWh * Cdyn_per_MWh)

    
    def render(self):
        """Render the environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass


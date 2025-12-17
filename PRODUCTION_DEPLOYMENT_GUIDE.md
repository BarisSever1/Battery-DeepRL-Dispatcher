# Production Deployment Guide: From Research to Real-World Day-Ahead Scheduling

## Executive Summary

This guide explains how to convert your RL-based battery scheduling system from a research project (using perfect information) into a production system that works with **forecasts** and generates **day-ahead schedules**. The key challenge is teaching your agent to make all 24 decisions at once (before the day starts) using predicted prices and PV generation, rather than making decisions hour-by-hour with perfect knowledge.

---

## The Core Problem: Perfect Information vs. Real Life

### Current System (Research)
- **Agent sees**: Actual prices and PV generation for all 24 hours at each step
- **Decision timing**: Hour-by-hour during the day
- **Information**: Perfect knowledge of future

### Real-World Requirements
- **Agent sees**: Forecasts for prices and PV (available day-before)
- **Decision timing**: All 24 decisions made before day starts (day-ahead)
- **Information**: Predictions with uncertainty

**Key Insight**: Your agent needs to learn to work with **forecasts** and make **all decisions upfront**, not reactively hour-by-hour.

---

## Part 1: Integrating Forecasts into Your System

### Understanding Forecast Sources

**What You Can Buy:**
- **Day-ahead energy price forecasts**: Available from market operators, energy analytics companies, or forecasting services
- **PV generation forecasts**: Available from weather services (Solcast, PVGIS, OpenWeatherMap) that convert weather forecasts to solar power
- **Reserve price forecasts**: You'll need to build this (see below)

**Forecast Format:**
Each forecast should provide 24 hourly values for tomorrow:
- `price_em_forecast[0..23]`: Predicted day-ahead energy prices (EUR/MWh)
- `p_res_total_forecast[0..23]`: Predicted PV generation (MW)
- `price_as_forecast[0..23]`: Predicted reserve prices (EUR/MW·h) - you build this

### How to Integrate Forecasts

**Step 1: Modify Your Environment to Accept Forecasts**

Currently, your environment loads actual historical data. You need to modify it so it can:
- Accept forecast data as input (instead of loading from historical files)
- Use forecasts to build the observation vector (same 19 dimensions, but with forecasted values)
- Still track actual performance for training/evaluation

**Conceptual Change:**
```
Current: env.reset() → loads actual data for day D
New:     env.reset(forecast_data=forecasts) → uses forecasts for day D
```

The observation vector stays the same structure (19 dimensions), but the values come from forecasts:
- `price_em` → `price_em_forecast[hour]`
- `p_res_total` → `p_res_total_forecast[hour]`
- `price_as` → `price_as_forecast[hour]`

Daily context features (max prices, min prices, etc.) are computed from the **forecasted** day, not actual.

**Step 2: Build Reserve Price Forecast**

Since you can't buy reserve price forecasts, you need to build your own. Options:

**Option A: Simple Correlation Model**
- Reserve prices often correlate with day-ahead energy prices
- Build a simple model: `price_as_forecast = f(price_em_forecast, historical_correlation)`
- Use historical data to learn the relationship

**Option B: Time-Series Forecast**
- Train a forecasting model (LSTM, XGBoost) on historical reserve prices
- Use features: historical reserve prices, day-ahead energy prices, day of week, season
- Similar approach to what you'd use for energy prices if you were building that

**Option C: Use Day-Ahead Energy Price as Proxy**
- If reserve prices are highly correlated, use: `price_as_forecast ≈ α * price_em_forecast + β`
- Calibrate α and β from historical data

**Recommendation**: Start with Option A or C (simpler), then move to Option B if needed.

---

## Part 2: Teaching Your Agent Day-Ahead Scheduling

### The Learning Challenge

**Good News**: You don't need to change your agent architecture! 

Your agent currently makes decisions **hour-by-hour during the day** with perfect information. For day-ahead scheduling, you just need to:
- Run it **before the day starts** (not during)
- Feed it **forecasts** (not actuals)
- Collect the 24 actions as your schedule

The agent still makes decisions hour-by-hour, but now it does so **in advance** using forecasted data.

### Training Strategy: Simple Hour-by-Hour with Forecasts

**Key Insight**: You don't need to change your agent architecture! Just run it **before the day starts** using **forecasts** instead of during the day with perfect information.

**Stage 1: Forecast-Aware Training**

Train your agent to work with forecasts instead of perfect information:

1. **For each training day:**
   - **Best approach**: Load historical forecasts for that day (forecasts that were made day-before)
     - Example: For training day 2024-03-15, use the forecast that was made on 2024-03-14
     - This gives you the exact forecast quality your agent will face in production
   - **Fallback**: If historical forecasts unavailable, use actuals as "perfect forecasts" (less realistic)
   - Agent runs hour-by-hour **before the day starts**, using forecast-based observations
   - Collect all 24 actions → this is your day-ahead schedule
   - Execute schedule with **actual** data
   - Compute reward based on **actual** performance
   - Update agent

**Key Insight**: By using historical forecasts vs. actuals, your agent learns to handle the exact forecast errors it will see in production. This is much better than simulating errors.

2. **Key Training Loop Change:**
   ```
   Current (Research): 
   # During the day, with perfect information
   for hour in 0..23:
       obs = env.get_observation(hour)  # with actual data
       action = agent.act(obs)
       reward = env.step(action)  # uses actual data
       agent.learn(obs, action, reward)
   
   New (Production Training):
   # Before the day starts, with forecasts
   historical_forecast = load_historical_forecast(day)  # forecast made on day-1
   env.reset(forecast_data=historical_forecast)  # Load forecast-based day
   
   schedule = []
   for hour in range(24):
       obs = env.get_observation(hour, forecast=True)  # uses forecast[hour]
       action = agent.act(obs)  # TD3 model outputs action for this hour
       schedule.append(action)
       env.step(action, forecast=True)  # advance simulated state using forecast
   
   # Now evaluate schedule with actual data
   actual_performance = env.execute_schedule(schedule, actual_data)
   reward = actual_performance['profit']
   # Learn from the full episode
   agent.learn_from_episode(obs_sequence, schedule, reward)
   ```

**Why This Works**: 
- **No agent architecture changes needed** - your existing hour-by-hour agent works perfectly!
- Agent still makes decisions step-by-step, just **before the day starts** instead of during
- Battery SOC and constraints are simulated hour-by-hour using forecasts, so physics are respected
- Agent learns from the mismatch between forecasts and reality
- The collected `schedule` (24 actions) is your day-ahead plan

3. **LSTM Advantage**: Your LSTM architecture is perfect for this! It processes the sequence and learns temporal patterns from forecasts, even when run hour-by-hour.

**Stage 2: Robustness Training**

Train agent to handle forecast uncertainty:

1. **Use Historical Forecast Errors**: If you have historical forecasts:
   - Compare past forecasts vs. actuals to analyze error patterns
   - Train agent on days with varying forecast quality (good forecasts, bad forecasts)
   - Agent naturally learns to handle the forecast errors it will see in production
   - This is more realistic than simulated errors

2. **Forecast Error Analysis**: Analyze your historical forecast data:
   - Compute forecast error statistics (MAE, RMSE, MAPE)
   - Identify patterns: Are forecasts worse on certain days? (weekends, holidays, extreme weather)
   - Use this to understand what errors agent needs to handle

3. **Multiple Scenarios** (if historical forecasts available):
   - Use ensemble forecasts (if multiple forecast sources available)
   - Or use forecast + confidence intervals (if forecast service provides uncertainty)
   - Train agent to perform well across forecast scenarios

4. **Uncertainty as Feature**: If forecast service provides confidence/uncertainty:
   - Add forecast confidence to observation
   - Agent learns: high uncertainty → be conservative, low uncertainty → be aggressive

### How the Agent Learns Day-Ahead Patterns

**What the Agent Needs to Learn:**

1. **Forecast Interpretation**: 
   - "If forecast says prices will be high at hour 18, I should discharge then"
   - "If forecast says PV will be high at hour 12, I should charge then"

2. **Uncertainty Handling**:
   - "If forecast is uncertain, be conservative (don't commit too much)"
   - "If forecast is confident, be aggressive (optimize more)"

3. **Temporal Coordination**:
   - "If I need to discharge at hour 18, I should charge before that"
   - "I need to manage SOC across the full day, not just hour-by-hour"

4. **Forecast Error Resilience**:
   - "Even if forecast is wrong, my schedule should still be reasonable"
   - "Don't over-optimize to forecast (robustness > perfect optimization)"

**How LSTM Helps:**

Your LSTM processes the full 24-hour sequence, so it can:
- See the full forecast pattern (all 24 hours at once)
- Learn temporal dependencies (charge before discharge)
- Learn to coordinate actions across the day
- Build internal representations of "good forecast patterns" vs. "bad forecast patterns"

**Training Data Requirements:**

- **Best Approach: Use Historical Forecasts** ✅
  - Request historical forecast data from your forecast providers
  - Match historical forecasts to actual data dates
  - Example: For training day 2024-03-15, use:
    - Forecast made on 2024-03-14 (for day 2024-03-15)
    - Actual data for 2024-03-15
  - This gives you the **exact forecast quality** your agent will face in production
  - Agent learns to handle real forecast errors, not simulated ones
  
- **How to Get Historical Forecasts**:
  - Ask your forecast service provider if they have historical data
  - Many services keep historical forecasts (for their own validation)
  - If unavailable, you might need to collect them going forward (start saving forecasts now)
  
- **If Historical Forecasts Unavailable** (fallback):
  - Use actuals as "perfect forecasts" (agent sees actual data)
  - Less realistic, but still trains agent on day-ahead scheduling structure
  - Or simulate by adding realistic noise to actuals (last resort)
  - But real historical forecasts are much better!

---

## Part 3: Applying It in Real Life

### Daily Workflow

**Day D-1 (Day Before Operating Day):**

1. **12:00-13:00**: Get forecasts for Day D
   - Fetch day-ahead price forecasts (from service) → `price_forecast[0..23]`
   - Fetch PV generation forecasts (from weather service) → `pv_forecast[0..23]`
   - Generate reserve price forecasts (using your model) → `reserve_forecast[0..23]`

2. **13:00-13:30**: Generate schedule
   - Load trained agent
   - Initialize environment with forecast data
   - **Run agent hour-by-hour** (before day starts):
     ```python
     schedule = []
     env.reset(forecast_data=forecasts)  # Load forecast-based day
     
     for hour in range(24):
         obs = env.get_observation(hour, forecast=True)  # uses forecast[hour]
         action = agent.act(obs, deterministic=True)  # deterministic for production
         schedule.append(action)
         env.step(action, forecast=True)  # advance simulated state
     ```
   - `schedule` now contains 24 actions (one per hour) - your day-ahead plan
   - Validate schedule (SOC constraints, power limits)

3. **13:30-14:00**: Submit to markets
   - Format schedule according to market requirements
   - Submit to day-ahead market
   - Submit reserve capacity offers

**Day D (Operating Day):**

1. **00:00-23:00**: Execute schedule
   - Battery controller follows pre-generated schedule hour-by-hour
   - Hour 0: Execute `schedule[0]`
   - Hour 1: Execute `schedule[1]`
   - ... and so on
   - Monitor actual vs. planned performance
   - Optional: Real-time adjustments (if market allows and forecast error is large)

### System Architecture

```
┌─────────────────┐
│ Forecast Service│ → Day-ahead prices, PV generation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reserve Forecaster│ → Reserve price forecasts (your model)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Trained Agent  │ → Generates 24-hour schedule
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Schedule Validator│ → Checks constraints
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Market Interface│ → Submits to markets
└─────────────────┘
```

### Real-Time Execution

**During the Operating Day:**

1. **Battery Controller**: Follows the pre-generated schedule hour-by-hour
   - Hour 0: Execute action[0]
   - Hour 1: Execute action[1]
   - ... and so on

2. **Monitoring**: Track actual performance
   - Actual PV vs. forecasted PV
   - Actual prices vs. forecasted prices
   - Actual SOC vs. planned SOC
   - Actual profit vs. expected profit

3. **Optional Adjustments**: If forecast error is large and market allows:
   - Re-optimize remaining hours with updated forecasts (intraday)
   - Update schedule (if market rules permit)

**Important**: Most markets have strict rules about schedule changes. Check your market's intraday adjustment policies.

### Performance Expectations

**Realistic Targets:**

- **Performance**: 70-85% of perfect-information performance
  - Perfect information = agent sees actual prices/PV (current research setup)
  - Forecast-based = agent sees forecasts (production setup)
  - 70-85% is typical for well-calibrated systems

- **Forecast Accuracy Requirements:**
  - Day-ahead prices: MAPE < 15% (Mean Absolute Percentage Error)
  - PV generation: RMSE < 20% of capacity
  - Reserve prices: Depends on your model quality

**Factors Affecting Performance:**

1. **Forecast Quality**: Better forecasts → better performance
2. **Market Volatility**: High volatility → harder to forecast → lower performance
3. **Agent Robustness**: How well agent handles uncertainty
4. **Battery Flexibility**: Larger battery → more room for error

### Handling Real-World Complications

**Problem 1: Forecast Service Down**

**Solution**: Fallback strategies
- Use previous day's schedule (if similar conditions)
- Use rule-based policy (from your baselines)
- Use conservative default (charge during cheap hours, discharge during expensive hours)

**Problem 2: Large Forecast Errors**

**Solution**: 
- Monitor forecast accuracy continuously
- If forecast error is consistently large, investigate forecast service
- Consider ensemble forecasts (average multiple sources)
- Train agent to be more robust to errors

**Problem 3: Market Rule Changes**

**Solution**:
- Keep environment configurable (YAML configs)
- Version control for market rules
- Regular review of market documentation
- Test suite that validates against current rules

**Problem 4: Agent Performance Degradation**

**Solution**:
- Continuous monitoring of daily profit
- Retrain agent periodically with recent data (every 3-6 months)
- A/B testing: Compare new agent version vs. current
- Online learning: Update agent with new experiences (advanced)

---

## Implementation Roadmap

### Phase 1: Forecast Integration (2-3 weeks)

**Goal**: System can use forecasts instead of actual data

1. Modify environment to accept forecast data as input
2. Integrate forecast services (APIs for prices and PV)
3. Build reserve price forecast model
4. **Collect historical forecasts** (if available from forecast services):
   - Request historical forecast data from your forecast providers
   - Match historical forecasts to actual data dates
   - This gives you realistic training data
5. Test with historical data (use historical forecasts if available, or actuals as "perfect forecasts" to verify)

### Phase 2: Day-Ahead Training (2-3 weeks)

**Goal**: Agent learns to work with forecasts (no architecture changes needed!)

1. Modify environment to accept forecast data
2. Modify training loop: Run agent hour-by-hour **before day starts** with forecasts
3. Collect 24 actions → evaluate as schedule with actual data
4. Train agent with historical forecasts (if available) or use actuals as "perfect forecasts"
5. Evaluate performance: forecast-based vs. perfect-information
6. Iterate: Improve agent robustness to forecast errors

**Key Advantage**: No need to change agent architecture - just change when it runs and what data it sees!

### Phase 3: Production System (2-3 weeks)

**Goal**: Automated daily scheduling service

1. Build scheduling service (runs daily, generates schedules)
2. Add schedule validation and error handling
3. Integrate with market submission (if automated)
4. Set up monitoring and logging

### Phase 4: Testing and Deployment (2-3 weeks)

**Goal**: Validate system works in production

1. Backtest on 6-12 months of historical data
2. Shadow mode: Generate schedules but don't submit (1-2 weeks)
3. Gradual rollout: Start with manual review, then automate
4. Monitor performance and iterate

**Total Timeline**: ~10-13 weeks for full implementation

---

## Key Success Factors

1. **Forecast Quality**: Invest in good forecast services. Poor forecasts = poor performance.

2. **Agent Robustness**: Train agent to handle forecast uncertainty, not just optimize to perfect forecasts.

3. **Validation**: Always validate schedules before submission (SOC constraints, power limits).

4. **Monitoring**: Track forecast accuracy and agent performance continuously.

5. **Fallbacks**: Always have backup strategies when forecasts fail or agent errors occur.

6. **Iteration**: System will improve over time as you collect more data and retrain agent.

---

## Summary: The Three Core Changes

1. **Forecasts Instead of Actuals**: Replace perfect information with forecasts (buy services for prices/PV, build model for reserves)

2. **Day-Ahead Scheduling**: Run agent hour-by-hour **before the day starts** using forecasts, collect 24 actions as schedule
   - **No agent architecture changes needed!**
   - Just change **when** it runs (before day vs. during day) and **what data** it sees (forecasts vs. actuals)

3. **Training for Uncertainty**: Train agent with historical forecasts vs. actuals, so it learns to handle forecast errors

**Key Advantage**: Your existing hour-by-hour agent works perfectly! You just:
- Run it before the day starts (not during)
- Feed it forecasts (not actuals)
- Collect the 24 actions as your day-ahead schedule

Your LSTM architecture is already well-suited for this! It processes sequences and learns temporal patterns, even when run hour-by-hour. The main work is:
- Integrating forecast data sources
- Modifying environment to accept forecasts
- Running agent before day starts instead of during
- Building the daily scheduling workflow

**Optional Future Improvement**: Later, you could train an agent to generate all 24 actions in one shot (sequence-to-sequence), but the hour-by-hour approach works great for now!

Good luck! 🚀

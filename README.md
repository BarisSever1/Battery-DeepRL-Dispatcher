# AI-Powered Battery Trading System

## What This Project Does

This project trains an artificial intelligence (AI) agent to operate a solar power plant with a battery storage system in the Hungarian electricity markets. The AI learns to make decisions every hour of the day about when to charge the battery, when to discharge it, and when to offer reserve capacity—all to maximize daily profit while respecting physical limits and accounting for battery wear.

Think of it like teaching a computer to be a professional energy trader, but one that can process thousands of scenarios instantly and learn from experience.

---

## How It Works: The Big Picture

### The Problem
Every day, you have:
- **Solar panels** generating electricity (varies by weather and time of day)
- **A battery** that can store energy (limited capacity)
- **Two markets** to trade in:
  - Day-ahead energy market (buy low, sell high)
  - Reserve market (get paid for being ready to help the grid)

The challenge: decide what to do each hour to make the most money, while:
- Never overcharging or over-discharging the battery
- Respecting power limits (can't charge/discharge too fast)
- Accounting for battery degradation (using the battery costs money over time)

### The Solution
An AI agent that:
1. **Learns from experience** by simulating thousands of trading days
2. **Remembers patterns** (e.g., "prices are usually high in the evening")
3. **Makes decisions** based on current conditions and learned strategies
4. **Improves over time** by learning what works and what doesn't

---

## Key Design Choices: What Makes This Different

This project uses a modified version of TD3 (Twin Delayed Deep Deterministic Policy Gradient), a state-of-the-art reinforcement learning algorithm. However, we made several important adaptations to make it work better for battery trading:

### 1. **Full Sequence Learning: Processing Entire Days at Once**

**Standard TD3 approach:**
- Processes one hour at a time
- Makes a decision, sees the result, then moves to the next hour
- Like reading a book one word at a time

**Our approach:**
- Processes the entire 24-hour day as one complete sequence
- The AI sees all 24 hours together and learns from the full day's pattern
- Like reading a whole chapter at once to understand the story better

**Why this matters:**
- Battery trading requires planning ahead (charge in the morning to sell in the evening)
- Seeing the full day helps the AI understand daily patterns (morning prices vs. evening prices)
- The AI can learn long-term strategies, not just hour-by-hour reactions

### 2. **Solving the Zero Padding Problem**

**The problem:**
When you process sequences of different lengths, you often need to "pad" shorter sequences with zeros to make them all the same size. This is like filling a short essay with blank lines to match a longer one. The problem: the AI might learn that zeros are meaningful, which confuses it.

**Our solution:**
We designed the memory system (replay buffer) to **never sample sequences that cross day boundaries**. This means:
- Every sequence is always exactly 24 hours (one complete day)
- No sequences are shorter than 24 hours
- No need for zero padding at all

**How we do it:**
- When storing experiences, we mark where each day ends
- When sampling for training, we check that the entire 24-hour sequence belongs to the same day
- If a sequence would cross into a different day, we skip it and pick another one

**Result:** The AI always sees complete, meaningful sequences without any confusing padding.

### 3. **No Discounting: Every Hour Matters Equally (Gamma = 1.0)**

**Standard approach:**
Most AI agents use "discounting" (gamma < 1.0), which means:
- Rewards in the near future are worth more than rewards far in the future
- Like preferring $100 today over $100 next year

**Our approach:**
We set gamma = 1.0, meaning:
- All hours of the day are equally important
- A profit at 8 AM is just as valuable as a profit at 8 PM
- The goal is to maximize the **total daily profit**, not just early profits

**Why this makes sense:**
- Battery trading is a daily cycle: you start and end each day
- There's no "future beyond the day" to discount
- We want the AI to optimize the entire day, not favor early hours

### 4. **Learning from All Time Steps, Not Just the Last One**

**Standard approach:**
Many sequence-based methods only learn from the final step of a sequence.

**Our approach:**
The AI learns from **every hour** in the 24-hour sequence:
- It computes how good each decision was at each hour
- It learns from all 24 hours simultaneously
- This makes training much more efficient

**Why this helps:**
- More learning signals per training batch
- Better understanding of how early decisions affect later outcomes
- Faster convergence to good strategies

---

## The Components

### 1. **The Simulator** (`src/envs/env_pv_bess.py`)
A realistic simulation of a trading day:
- 24 decision points (one per hour)
- Real market prices and solar generation data
- Enforces all physical limits (battery capacity, power limits, etc.)
- Calculates profits, revenues, and battery degradation costs

### 2. **The AI Brain** (`src/rl/models.py`)
Neural networks that make decisions:
- **Actor network**: Decides what action to take (charge, discharge, or idle)
- **Twin Critic networks**: Evaluate how good each decision is (like having two judges to avoid bias)
- **LSTM memory**: Remembers what happened earlier in the day to make better decisions

### 3. **The Memory System** (`src/rl/buffer.py`)
Stores past experiences:
- Remembers thousands of past trading days
- Samples random days for training (breaks patterns to avoid overfitting)
- Ensures sequences never cross day boundaries (solves the zero padding problem)

### 4. **The Learning Algorithm** (`src/rl/td3.py`)
The TD3 algorithm with our modifications:
- Processes full 24-hour sequences
- Learns from all time steps
- Uses gamma = 1.0 (no discounting)
- Twin critics for stability
- Delayed policy updates for better learning

### 5. **Training Script** (`scripts/train_td3.py`)
Orchestrates the learning process:
- Runs simulations day after day
- Collects experiences
- Updates the AI's knowledge
- Saves checkpoints and logs progress

---

## How Training Works

1. **Warmup Phase**: The AI takes random actions to explore and fill its memory
2. **Learning Phase**: 
   - The AI acts based on its current knowledge
   - Experiences are stored in memory
   - Random batches of complete 24-hour sequences are sampled
   - The AI learns from these sequences to improve its strategy
3. **Evaluation**: Periodically, the AI is tested on new days it hasn't seen
4. **Repeat**: This continues for many epochs until the AI becomes skilled

---

## Configuration Files

### `config/limits.yaml`
Physical and operational limits:
- Battery capacity (MWh)
- Maximum charge/discharge power (MW)
- Efficiency losses
- Grid connection limits

### `config/features.yaml`
What information the AI sees:
- Market prices
- Solar generation forecasts
- Time of day indicators
- Battery state of charge
- And more...

### `config/training.yaml`
Training hyperparameters:
- Learning rates
- Sequence length (24 hours)
- Batch sizes
- Exploration noise
- And more...

---

## Key Differences from Standard TD3

| Feature | Standard TD3 | Our Implementation |
|---------|--------------|-------------------|
| **Input** | Single time step | Full 24-hour sequence |
| **Learning** | Last step only | All 24 time steps |
| **Discounting** | Gamma < 1.0 | Gamma = 1.0 (no discount) |
| **Padding** | Often needed | Never needed (boundary checking) |
| **Context** | Limited | Full daily context via LSTM |

---

## Results and Evaluation

The trained agent is evaluated against baseline strategies:
- **PV+DS3 (Reserve Only)**: Only participates in reserve market
- **PV+DA (Two Cycles)**: Simple buy-low, sell-high strategy
- **Realistic Trader**: Human-like threshold-based strategy

The AI typically outperforms these baselines by:
- Better timing of charge/discharge cycles
- More sophisticated reserve market participation
- Optimizing for total daily profit, not just individual trades

---

## Technical Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas
- Gymnasium
- See `requirements.txt` for full list

---

## Quick Start: Training and Evaluation

### Training a New Agent

Train a new agent from scratch with default settings:

```bash
python scripts/train_td3.py \
  --checkpoint-dir checkpoints/my_run \
  --csv-log checkpoints/my_run/train_log.csv \
  --quick-eval-interval 100 \
  --quick-eval-outdir results/quick/my_run
```

**What this does:**
- `--checkpoint-dir`: Where to save model checkpoints (saved every 100 episodes by default)
- `--csv-log`: Path to CSV file for logging episode metrics (profit, revenue, degradation)
- `--quick-eval-interval`: Run quick evaluation every N episodes (100 = every 100 episodes)
- `--quick-eval-outdir`: Directory to save quick evaluation results (hourly data for each season)

**Common training options:**
- `--epochs 100`: Number of training epochs (default: 100)
- `--days-per-epoch 22`: Number of training days per epoch (default: auto-detected from data)
- `--batch-size 128`: Mini-batch size for training (default: 128)
- `--lr 2e-4`: Learning rate (default: from config/training.yaml)
- `--seed 42`: Random seed for reproducibility

**Example with custom parameters:**
```bash
python scripts/train_td3.py \
  --checkpoint-dir checkpoints/td3_experiment1 \
  --epochs 200 \
  --batch-size 256 \
  --lr 1e-4 \
  --seed 123 \
  --csv-log checkpoints/td3_experiment1/train_log.csv \
  --quick-eval-interval 50 \
  --quick-eval-outdir results/quick/td3_experiment1
```

### Evaluating a Trained Agent

Run comprehensive evaluation on a trained checkpoint:

```bash
python scripts/eval_comprehensive.py \
  --actor-checkpoint checkpoints/td3_dec11_gyongosh1/actor_ep1000.pth \
  --output-dir results/eval_td3_dec11_gyongosh1
```

**What this does:**
- `--actor-checkpoint`: Path to the trained actor network checkpoint (`.pth` file)
- `--output-dir`: Directory to save evaluation results (hourly data, daily summaries, comparisons with baselines)

**What you get:**
- Hourly operational data for all test dates
- Daily profit/revenue summaries
- Comparison with baseline strategies (PV+DS3, PV+DA, Energiabőrze)
- Performance metrics and statistics

**Note:** The checkpoint path should point to the `actor_epXXXX.pth` file. The corresponding critic checkpoint should be in the same directory with the same episode number.

### Complete Example Workflow

1. **Train a new agent:**
```bash
python scripts/train_td3.py \
  --checkpoint-dir checkpoints/td3_dec16_v1 \
  --csv-log checkpoints/td3_dec16_v1/train_log.csv \
  --quick-eval-interval 100 \
  --quick-eval-outdir results/quick/td3_dec16_v1
```

2. **Wait for training to complete or stop it early** (checkpoints are saved periodically)

3. **Evaluate the best checkpoint:**
```bash
python scripts/eval_comprehensive.py \
  --actor-checkpoint checkpoints/td3_dec16_v1/actor_ep1000.pth \
  --output-dir results/eval_td3_dec16_v1
```

4. **View results:**
- Training metrics: `results/quick/td3_dec16_v1/training_metrics.jsonl`
- Evaluation results: `results/eval_td3_dec16_v1/hourly_data.csv`
- Checkpoints: `checkpoints/td3_dec16_v1/actor_epXXXX.pth`

---

## Why These Design Choices Matter

### For Battery Trading Specifically:
- **Full sequences**: Battery decisions are interdependent—what you do at 8 AM affects what you can do at 8 PM
- **No discounting**: In a daily cycle, all hours contribute equally to profit
- **No padding**: Clean data means the AI learns real patterns, not artificial zeros
- **All time steps**: More efficient learning from limited training data

### For AI Stability:
- **Boundary checking**: Prevents the AI from learning meaningless patterns from padding
- **Complete sequences**: The LSTM memory works better with complete, meaningful sequences
- **Gamma = 1.0**: Simpler objective function reduces training complexity

---

## Future Improvements

- Integration with forecast data (instead of perfect information)
- Multi-day planning (extending beyond 24 hours)
- Real-time market adaptation
- Risk-aware trading strategies

---

## References

- TD3 Algorithm: "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)
- HUPX Market: https://hupx.hu/
- MAVIR (Hungarian TSO): https://www.mavir.hu/

---

## License

MIT License — see `LICENSE` file

---

**Status**: Active development

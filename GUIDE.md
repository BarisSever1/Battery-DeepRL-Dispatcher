# Complete Training Process Guide

## Overview

This guide explains exactly how the AI training process works, step by step, from the moment you run the command until the AI finishes learning. Every file, every function, and every step is explained in simple terms.

---

## Where It All Starts

**File:** `scripts/train_td3.py`

This is the main entry point. When you run:
```bash
python scripts/train_td3.py --checkpoint-dir checkpoints/my_run --epochs 100
```

Python starts executing code from this file, beginning with the `main()` function at the bottom of the file.

---

## Step 1: Reading Your Command (parse_args function)

**Location:** Lines 27-80 in `scripts/train_td3.py`

**What it does:**
- Reads all the command-line arguments you provided (like `--epochs`, `--checkpoint-dir`, etc.)
- Sets default values for anything you didn't specify
- Returns a structured object containing all these settings

**In simple terms:** It's like a receptionist taking your order and writing down all your preferences before the kitchen (training process) starts cooking.

---

## Step 2: Setting Up the Kitchen (main function - Setup Phase)

**Location:** Lines 382-500 in `scripts/train_td3.py`

The `main()` function is like the head chef organizing everything before cooking begins.

### 2.1 Loading Configuration Files

**What happens:**
- Reads `config/training.yaml` to get training settings
- Reads `config/features.yaml` to understand what data features to use
- Reads `config/limits.yaml` to know the physical constraints

**Why:** These files contain all the "recipes" and "rules" the system needs to follow.

### 2.2 Creating the Simulator (Environment)

**Code:** `env = BESSEnv(data_path=args.data_path, degradation_model=args.degradation_model)`

**Location:** Line 410 in `scripts/train_td3.py`

**What it does:**
- Creates a virtual trading day simulator
- Loads historical market data (prices, solar generation)
- Sets up all the rules (battery limits, power constraints, etc.)

**In simple terms:** This is like building a flight simulator—a safe place to practice before flying a real plane. The AI will practice trading here thousands of times.

**Key functions in `src/envs/env_pv_bess.py`:**
- `__init__()`: Sets up the simulator with all its rules
- `reset()`: Starts a new trading day (resets battery to initial state, picks a new date)
- `step()`: Executes one hour of trading (takes action, calculates profit, updates battery state)

### 2.3 Creating the AI Agent

**Code:** `agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, config=config, device=device)`

**Location:** Lines 452-457 in `scripts/train_td3.py`

**What it does:**
- Creates the "brain" that will make trading decisions
- Initializes two neural networks:
  - **Actor network**: The decision-maker (decides what to do)
  - **Critic networks**: Two evaluators (judge how good decisions are)

**In simple terms:** This is like hiring a new trader and giving them a blank notebook. They don't know anything yet—they'll learn through experience.

**Key functions in `src/rl/td3.py`:**
- `__init__()`: Creates and initializes all the neural networks
- `act()`: Makes a decision based on current situation (uses Actor network)
- `update()`: Learns from past experiences (updates both Actor and Critic networks)

### 2.4 Creating the Memory System (Replay Buffer)

**Code:** `buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=args.buffer_size, n_step=config.n_step, gamma=config.gamma)`

**Location:** Lines 473-479 in `scripts/train_td3.py`

**What it does:**
- Creates a large storage system to remember past trading experiences
- Can store up to 200,000 (or whatever you specify) individual trading hours
- Each memory contains: what the situation was, what action was taken, what reward was received, what happened next

**In simple terms:** This is like a massive filing cabinet where the AI stores every trading hour it experiences, so it can review them later to learn.

**Key functions in `src/rl/buffer.py`:**
- `add()`: Stores a new experience (one hour of trading)
- `sample()`: Randomly picks a batch of complete 24-hour sequences for learning
- `_is_valid_index()`: Ensures sequences never cross day boundaries (solves zero padding problem)

### 2.5 Setting Up Logging and Checkpoints

**Code:** Lines 483-493 in `scripts/train_td3.py`

**What it does:**
- Creates directories to save progress
- Sets up CSV files to log metrics
- Prepares to save model checkpoints periodically

**In simple terms:** Setting up the recording equipment to track how well the AI is learning.

---

## Step 3: The Training Loop (main function - Training Phase)

**Location:** Lines 500-780 in `scripts/train_td3.py`

This is where the actual learning happens. The loop has three nested levels:

### Level 1: Epochs (Outer Loop)
```python
for epoch in range(args.epochs):  # e.g., 100 epochs
```

**What it means:** One epoch = one complete pass through all training days. If you have 22 training days and 100 epochs, the AI will see each day 100 times.

### Level 2: Episodes/Days (Middle Loop)
```python
for day_idx in range(args.days_per_epoch):  # e.g., 22 days per epoch
```

**What it means:** One episode = one complete trading day (24 hours). The AI will practice trading one full day, then move to the next day.

### Level 3: Hours/Steps (Inner Loop)
```python
for hour in range(24):  # 24 hours in a day
```

**What it means:** One step = one hour of trading. The AI makes a decision, executes it, sees the result, and moves to the next hour.

---

## Step 4: One Complete Episode (Trading Day) - Detailed Walkthrough

Let's trace through exactly what happens during one complete trading day:

### 4.1 Starting a New Day

**Code:** `obs_vec, info = env.reset(options={'date': current_date})`

**Location:** Line 570 in `scripts/train_td3.py`

**What happens:**
1. The simulator picks a specific date from your training data
2. Resets the battery to its starting state (usually 50% charged)
3. Loads all the market data for that date (prices, solar generation for all 24 hours)
4. Returns the initial observation (the current situation the AI sees)

**In simple terms:** "Okay, it's 6:00 AM on January 15th. Battery is at 50%. Here's what the market looks like today. Ready to trade?"

**Behind the scenes in `src/envs/env_pv_bess.py`:**
- `reset()` function (lines 233-261):
  - Selects a date from available training dates
  - Loads that day's data (prices, PV generation, etc.)
  - Resets battery state of charge to initial value
  - Sets current hour to 0
  - Returns the first observation

### 4.2 The 24-Hour Loop

For each hour from 0 to 23:

#### Step A: The AI Makes a Decision

**Code:** `action = agent.act(state_seq, eval_mode=False)`

**Location:** Line 580 in `scripts/train_td3.py`

**What happens:**
1. The AI looks at the current situation (observation)
2. The Actor network processes this information
3. The Actor outputs a number between 0 and 1 (the action)
4. Exploration noise is added (to encourage trying new things)
5. The action is returned

**In simple terms:** The AI thinks "Based on what I see, I should charge the battery at 70% capacity this hour."

**Behind the scenes in `src/rl/td3.py`:**
- `act()` function (lines 115-180):
  - Takes the current state sequence
  - Passes it through the Actor LSTM network
  - The LSTM processes the sequence and outputs an action
  - Adds exploration noise (randomness) to encourage exploration
  - Clamps the action to valid range [0, 1]

**Behind the scenes in `src/rl/models.py`:**
- `ActorLSTM.forward()` (lines 81-110):
  - Takes state sequence [batch, time, features]
  - LSTM processes the sequence, remembering earlier hours
  - MLP layers make the final decision
  - Outputs action value

#### Step B: Executing the Decision

**Code:** `next_obs_vec, reward, terminated, truncated, step_info = env.step(action)`

**Location:** Line 581 in `scripts/train_td3.py`

**What happens:**
1. The simulator takes the action (e.g., "charge at 70%")
2. Calculates what actually happens:
   - How much energy flows
   - What the new battery state is
   - How much money was made/lost
   - Whether any limits were violated
3. Calculates the reward (profit for this hour)
4. Moves to the next hour
5. Returns the new situation and results

**In simple terms:** "Okay, you wanted to charge at 70%. Here's what actually happened: you charged 0.7 MW, made €5.20, battery is now at 52%. It's now 7:00 AM."

**Behind the scenes in `src/envs/env_pv_bess.py`:**
- `step()` function (lines 524-563):
  - Interprets the action (positive = discharge in peak hours, charge in off-peak)
  - Applies physical constraints (can't exceed battery capacity, power limits, etc.)
  - Calculates energy flows (battery, grid, PV)
  - Computes revenues (energy trading, reserves)
  - Computes degradation cost
  - Calculates reward = revenue - degradation
  - Updates battery state of charge
  - Moves to next hour
  - Returns new observation, reward, and info

#### Step C: Storing the Experience

**Code:** `buffer.add(obs_vec, action, reward, next_obs_vec, done)`

**Location:** Line 600 in `scripts/train_td3.py`

**What happens:**
1. Takes the complete experience: (old situation, action taken, reward received, new situation, whether day ended)
2. Stores it in the replay buffer memory
3. The memory now has one more trading hour to learn from later

**In simple terms:** "Write this down in the notebook: At 6 AM, situation was X, I did Y, got reward Z, new situation is W."

**Behind the scenes in `src/rl/buffer.py`:**
- `add()` function (lines 80-106):
  - Stores state, action, reward, next_state, done in pre-allocated arrays
  - Uses circular buffer (when full, overwrites oldest memories)
  - Updates position and size counters

#### Step D: Learning from Past Experiences (If Buffer is Full Enough)

**Code:** 
```python
if len(buffer) >= args.warmup_steps:
    for _ in range(args.updates_per_step):
        stats = agent.update(buffer, batch_size=args.batch_size, seq_len=args.seq_len)
```

**Location:** Lines 609-628 in `scripts/train_td3.py`

**What happens:**
1. Checks if the memory has enough experiences (usually needs 5,000+ hours)
2. If yes, randomly samples a batch of complete 24-hour sequences from memory
3. The AI learns from these sequences to improve its strategy
4. This happens every hour once the buffer is full enough

**In simple terms:** "I've seen enough trading hours. Let me review some random past days to figure out what works and what doesn't."

**Behind the scenes in `src/rl/buffer.py`:**
- `sample()` function (lines 108-194):
  - Randomly picks batch_size number of end-indices
  - For each index, gathers a complete 24-hour sequence ending at that index
  - Ensures sequences never cross day boundaries (checks `_is_valid_index()`)
  - Returns batch of sequences: states [batch, 24, features], actions [batch, 24, 1], etc.

**Behind the scenes in `src/rl/td3.py`:**
- `update()` function (lines 200-345):
  - **Step 1:** Samples batch of 24-hour sequences from buffer
  - **Step 2:** Computes target Q-values for ALL 24 time steps:
    - Uses target networks (stale copies of main networks for stability)
    - For each hour: target = reward + gamma * (next Q-value)
    - Since gamma=1.0, this is: target = reward + next Q-value
  - **Step 3:** Updates Critic networks:
    - Current critics predict Q-values for all 24 hours
    - Compares predictions to targets
    - Computes loss (how wrong the predictions were)
    - Backpropagates error to improve critics
  - **Step 4:** Updates Actor network (every policy_delay steps):
    - Actor suggests actions for all 24 hours
    - Critics evaluate how good those actions are
    - Actor is updated to prefer actions that get higher Q-values
  - Returns statistics (losses, Q-values, etc.)

### 4.3 Ending the Day

**Code:** Lines 630-671 in `scripts/train_td3.py`

**What happens:**
1. The 24-hour loop completes
2. Total daily profit is calculated
3. Metrics are logged to CSV file
4. Training metrics (losses, Q-values) are saved to JSONL file
5. Progress is printed to console

**In simple terms:** "Day complete! Total profit: €45.20. Revenue: €50.00, Degradation cost: €4.80. Let me write this down and move to the next day."

---

## Step 5: Periodic Checkpoints and Evaluations

**Location:** Lines 683-780 in `scripts/train_td3.py`

### 5.1 Saving Checkpoints

**Code:** Lines 683-702

**What happens:**
Every N episodes (e.g., every 100 episodes):
1. Saves the current Actor network weights to a file
2. Saves the current Critic network weights to a file
3. Saves metadata (episode number, average return) to JSON file

**Why:** So you can stop training and resume later, or use a specific checkpoint for evaluation.

**In simple terms:** "Taking a snapshot of the AI's current knowledge so we can come back to it later."

### 5.2 Quick Evaluations

**Code:** Lines 704-780

**What happens:**
Periodically (e.g., every 500 episodes):
1. Creates a fresh environment (not used for training)
2. Picks one date from each season (winter, spring, summer, fall)
3. Runs the current AI policy on these dates (no exploration, pure decision-making)
4. Saves detailed hourly results to CSV files

**Why:** To see how well the AI is performing on days it hasn't trained on, and to track progress over time.

**In simple terms:** "Let me test the AI on some new days to see how good it's getting."

**Behind the scenes:**
- `quick_rollout()` function (lines 216-370 in `scripts/train_td3.py`):
  - Resets environment to a specific date
  - Runs 24 hours with current policy (no exploration)
  - Collects all hourly data (SOC, actions, prices, profits, etc.)
  - Returns list of dictionaries with all the data

---

## Step 6: End of Training

**Location:** Lines 780-805 in `scripts/train_td3.py`

**What happens:**
1. All epochs complete
2. Final checkpoint is saved
3. CSV files are closed
4. Summary statistics are printed
5. Training is complete!

**In simple terms:** "Training finished! The AI has learned from thousands of trading days. Final model saved."

---

## Data Flow Summary

Here's how data flows through the system during training:

```
1. You run: python scripts/train_td3.py
   ↓
2. parse_args() reads your command
   ↓
3. main() sets up:
   - Environment (simulator)
   - Agent (AI brain)
   - Buffer (memory)
   ↓
4. Training loop starts:
   For each epoch (100 times):
     For each day (22 days):
       env.reset() → starts new day
       For each hour (24 hours):
         agent.act() → makes decision
         env.step() → executes, returns reward
         buffer.add() → stores experience
         if buffer full enough:
           buffer.sample() → gets random sequences
           agent.update() → learns from sequences
       Log metrics, save checkpoint
   ↓
5. Training complete, models saved
```

---

## Key Functions Reference

### In `scripts/train_td3.py`:

- **`parse_args()`**: Reads command-line arguments
- **`main()`**: Orchestrates entire training process
- **`initialize_csv()`**: Sets up logging file
- **`get_one_date_per_season()`**: Picks evaluation dates
- **`quick_rollout()`**: Tests AI on specific dates
- **`guided_warmup_action()`**: Smart random actions during warmup

### In `src/envs/env_pv_bess.py`:

- **`__init__()`**: Creates simulator, loads data
- **`reset()`**: Starts new trading day
- **`step()`**: Executes one hour, returns results
- **`_get_observation()`**: Builds the state vector the AI sees
- **`_apply_constraints()`**: Enforces physical limits
- **`_calculate_degradation_cost()`**: Computes battery wear cost

### In `src/rl/td3.py`:

- **`__init__()`**: Creates Actor and Critic networks
- **`act()`**: Makes decision based on current state
- **`update()`**: Learns from batch of sequences
- **`compute_q_at_episode_start()`**: Computes Q-values at day start

### In `src/rl/models.py`:

- **`ActorLSTM`**: Neural network that makes decisions
- **`CriticLSTM`**: Neural networks that evaluate decisions
- **`forward()`**: Processes sequences through LSTM and MLP layers

### In `src/rl/buffer.py`:

- **`__init__()`**: Creates empty memory storage
- **`add()`**: Stores one experience
- **`sample()`**: Randomly samples batch of complete sequences
- **`_is_valid_index()`**: Ensures sequences don't cross day boundaries
- **`_gather_sequences()`**: Builds 24-hour sequences from memory

---

## Understanding the Learning Process

### Warmup Phase (First 5,000 steps)

- AI takes mostly random actions
- Goal: Fill memory with diverse experiences
- No learning happens yet (buffer not full enough)
- Like a new employee shadowing and observing before making decisions

### Learning Phase (After warmup)

- AI acts based on current knowledge (with some exploration)
- Every hour, if buffer is full:
  - Samples random batch of 24-hour sequences
  - Updates Critic networks (learns to predict Q-values)
  - Updates Actor network (learns better actions)
- Like an employee learning from past experiences to improve

### Convergence Phase (Later epochs)

- AI has learned good strategies
- Makes more consistent, profitable decisions
- Still explores occasionally to find improvements
- Like an experienced employee who knows the job well but still learns new tricks

---

## Important Concepts Explained Simply

### State (Observation)
A vector of numbers describing the current situation:
- Current hour (0-23)
- Battery state of charge (0-100%)
- Current prices
- Solar generation forecast
- Time until peak hours
- And more...

### Action
A single number between 0 and 1:
- 0 = idle (do nothing)
- 0.5 = moderate charge/discharge
- 1.0 = maximum charge/discharge
- The environment interprets this based on peak/non-peak hours

### Reward
Profit for one hour:
- Revenue from energy trading
- Revenue from reserves
- Minus degradation cost
- Normalized (divided by 200 to keep numbers manageable)

### Q-Value
The AI's estimate of "how good is this situation + action":
- High Q-value = this is a good decision, will lead to good profits
- Low Q-value = this is a bad decision, will lead to losses
- The Critic networks learn to predict Q-values accurately

### Loss
How wrong the AI's predictions are:
- Critic loss = how wrong Q-value predictions are
- Actor loss = how much the actor could improve
- Lower loss = better predictions = better decisions

---

## Troubleshooting: What to Check If Training Fails

1. **Check data path**: Is `--data-path` pointing to valid parquet file?
2. **Check buffer size**: Is it filling up? (Should see "Warmup Complete" message)
3. **Check losses**: Are they NaN or exploding? (Check training_metrics.jsonl)
4. **Check memory**: Are sequences being sampled? (Check that buffer.sample() works)
5. **Check device**: Is it using CPU/GPU as expected?

---

## Summary

The training process is a carefully orchestrated dance:

1. **Setup**: Create simulator, AI, and memory
2. **Loop**: For many epochs, for many days, for 24 hours each:
   - AI observes situation
   - AI makes decision
   - Simulator executes and returns reward
   - Experience is stored
   - AI learns from past experiences
3. **Monitor**: Log progress, save checkpoints, evaluate periodically
4. **Complete**: Save final model

Every function has a specific role, and they all work together to teach the AI how to trade batteries profitably. The key innovation is processing full 24-hour sequences and learning from all time steps, which makes the AI much better at planning ahead and understanding daily patterns.

---

**End of Guide**


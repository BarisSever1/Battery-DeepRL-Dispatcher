"""Twin Delayed DDPG (TD3) implementation with LSTM-based networks."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .buffer import ReplayBuffer
from .models import ActorLSTM, CriticLSTM


@dataclass
class TD3Config:
    """
    Simple container for all TD3 hyperparameters.

    You usually do not change this class – instead you pass different values
    from a config file (YAML) when constructing the agent.
    """

    gamma: float = 0.995
    n_step: int = 1
    use_n_step: bool = True
    tau: float = 0.01
    policy_delay: int = 2
    lr: float = 5e-5
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    exploration_std: float = 0.5
    target_std: float = 0.5
    noise_clip: float = 0.4
    action_low: float = -1.0
    action_high: float = 1.0


class TD3Agent:
    """
    Main TD3 agent class.

    It owns:
      * the actor network (policy),
      * two critic networks (Q‑functions),
      * their target copies,
      * and all training logic (update steps).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        mlp_hidden_sizes: Tuple[int, int] = (64, 64),
        config: Optional[TD3Config] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        # Move everything to the selected device (CPU or GPU)
        self.device = torch.device(device)
        # Remember basic problem dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        # If no config is passed in, fall back to default hyperparameters above
        self.config = config or TD3Config()

        # Parameters used to build the actor network
        actor_kwargs = {
            "state_dim": state_dim,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_layers,
            "mlp_hidden_sizes": mlp_hidden_sizes,
        }
        # Parameters used to build the critic networks
        critic_kwargs = {
            "state_dim": state_dim,
            "act_dim": action_dim,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_layers,
            "mlp_hidden_sizes": mlp_hidden_sizes,
        }

        # Online networks used for acting and learning
        self.actor = ActorLSTM(**actor_kwargs).to(self.device)
        self.critic = CriticLSTM(**critic_kwargs).to(self.device)
        # Target networks are slow‑moving copies for stable targets
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # Optimizers for gradient descent
        actor_lr = self.config.actor_lr or self.config.lr
        critic_lr = self.config.critic_lr or self.config.lr
        self.actor_optimizer: Optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer: Optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Counts how many gradient steps we took (used for delayed policy updates)
        self.total_it = 0

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(
        self,
        state_seq: np.ndarray,
        eval_mode: bool = False,
    ) -> np.ndarray:
        """
        Compute an action from a sequence of states.

        Args:
            state_seq: numpy array of shape [batch, time, features]. In practice
                       we pass batch=1 and the last N observations as history.
            eval_mode: if True → no exploration noise is added (pure policy).

        Returns:
            numpy array of shape [batch, action_dim] with values in [action_low, action_high].
        """
        # Put the actor into evaluation mode and disable gradients (no training here)
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state_seq, dtype=torch.float32, device=self.device)
            # ActorLSTM returns an action for every time step in the sequence
            action_seq, hidden = self.actor(state_tensor)
            # We do not use the hidden state outside, so just detach it
            hidden = tuple(h.detach() for h in hidden)

            if action_seq.dim() == 2:  # [B,A] -> [B,1,A]
                action_seq = action_seq.unsqueeze(1)

            # TD3 works on the last time step: use only the final action
            actions_last = action_seq[:, -1:, :]  # [B,1,A]

            if not eval_mode:
                # Add Gaussian exploration noise during training (unclipped for better exploration)
                noise = torch.randn_like(actions_last) * float(self.config.exploration_std)
                # Noise clipping removed to allow sign flips (e.g., -0.9 + noise can become positive)
                actions_last = actions_last + noise

            # Clamp action to the legal range
            actions_last = actions_last.clamp(self.config.action_low, self.config.action_high)
            actions_np = actions_last.squeeze(1).cpu().numpy()

        # Switch back to training mode for further updates
        self.actor.train()
        return actions_np

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, buffer: ReplayBuffer, batch_size: int, seq_len: int = 24) -> dict:
        """
        One TD3 update step using a minibatch of sequences from the replay buffer.

        This:
          1. Samples sequences of length `seq_len`.
          2. Computes target Q values using the target networks.
          3. Updates the critic to fit those targets.
          4. Every `policy_delay` steps, updates the actor using the critic.
        """
        self.total_it += 1

        # ---- 1) Sample a batch of sequences from replay ----
        batch = buffer.sample(batch_size, seq_len=seq_len)
        states_seq = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions_seq = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards_raw = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states_seq = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones_raw = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        seq_len_actual = states_seq.shape[1]

        def _last_step(t: Tensor) -> Tensor:
            """
            Helper: extract the last time step from a [B,T,*] tensor so we can
            work with a simple [B,1] shape for scalars like rewards/dones.
            """
            if t.dim() == 3:
                return t[:, -1, :].view(t.size(0), -1)
            if t.dim() == 2:
                return t[:, -1].view(t.size(0), 1)
            if t.dim() == 1:
                return t.view(t.size(0), 1)
            raise ValueError("Unexpected tensor rank")

        batch_size = states_seq.size(0)
        seq_len_actual = states_seq.shape[1]

        # Extract rewards and dones for all time steps
        # rewards_raw shape: [B, T] or [B, T, 1]
        if rewards_raw.dim() == 3:
            rewards_all = rewards_raw.squeeze(-1)  # [B, T]
        else:
            rewards_all = rewards_raw  # [B, T]
        
        if dones_raw.dim() == 3:
            dones_all = dones_raw.squeeze(-1)  # [B, T]
        else:
            dones_all = dones_raw  # [B, T]

        # For sequence returns: we learn Q-values for ALL time steps in the sequence
        # This better utilizes the temporal structure for battery dispatch problems
        actions_for_critic = actions_seq

        # ---- 2) Compute target Q values for ALL time steps (no gradients) ----
        with torch.no_grad():
            # Get next actions for all next states in the sequence
            # next_states_seq[:, t, :] is the next state after states_seq[:, t, :]
            next_actions_full, hidden_next = self.actor_target(next_states_seq)
            hidden_next = tuple(h.detach() for h in hidden_next)
            if next_actions_full.dim() == 2:
                next_actions_full = next_actions_full.unsqueeze(1)

            # TD3 target-policy smoothing: add noise to actions in the sequence
            # Primary smoothing on the last action (standard TD3), with smaller noise
            # on earlier steps to improve exploration across the action history.
            next_actions_seq = next_actions_full.clone()
            
            # Add noise to the last action (primary smoothing, standard TD3)
            next_actions_last = next_actions_full[:, -1:, :]
            target_noise = torch.randn_like(next_actions_last) * self.config.target_std
            target_noise = target_noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            noisy_last = (next_actions_last + target_noise).clamp(
                self.config.action_low, self.config.action_high
            )
            next_actions_seq[:, -1:, :] = noisy_last
            
            # Add smaller noise to earlier steps for better exploration (optional improvement)
            if next_actions_seq.shape[1] > 1:
                # Use 30% of target_std for earlier steps to maintain exploration without
                # destabilizing the target policy too much
                early_noise_std = self.config.target_std * 0.3
                early_noise = torch.randn_like(next_actions_seq[:, :-1, :]) * early_noise_std
                early_noise = early_noise.clamp(-self.config.noise_clip * 0.3, self.config.noise_clip * 0.3)
                next_actions_seq[:, :-1, :] = (
                    next_actions_seq[:, :-1, :] + early_noise
                ).clamp(self.config.action_low, self.config.action_high)

            # Compute Q-values for next states (all time steps)
            q1_next, q2_next = self.critic_target(next_states_seq, next_actions_seq)
            q1_next_hidden = tuple(h.detach() for h in q1_next.hidden)
            q2_next_hidden = tuple(h.detach() for h in q2_next.hidden)
            # Clipped double‑Q: take the smaller of the two target critics for all time steps
            min_q_next = torch.min(q1_next.outputs, q2_next.outputs)  # [B, T, 1]
            
            # Compute target Q for ALL time steps: r_t + γ * (1-done_t) * Q(s_{t+1}, a_{t+1})
            # For each time step t: target_q[t] = r[t] + γ * (1 - done[t]) * min_q_next[t]
            # rewards_all: [B, T], min_q_next: [B, T, 1]
            rewards_expanded = rewards_all.unsqueeze(-1)  # [B, T, 1]
            dones_expanded = dones_all.unsqueeze(-1)  # [B, T, 1]
            target_q_all = rewards_expanded + (1.0 - dones_expanded) * self.config.gamma * min_q_next  # [B, T, 1]

        # ---- 3) Critic update: fit Q(s_{1:T}, a_{1:T}) to target_q for ALL time steps ----
        q1_current, q2_current = self.critic(states_seq, actions_for_critic)
        # q1_current.outputs and q2_current.outputs have shape [B, T, 1]
        
        # Compute loss over ALL time steps in the sequence (not just the last one)
        # This better utilizes the temporal structure and learns from all rewards in the sequence
        q1_all = q1_current.outputs  # [B, T, 1]
        q2_all = q2_current.outputs  # [B, T, 1]
        
        critic_loss = nn.functional.mse_loss(q1_all, target_q_all) + nn.functional.mse_loss(q2_all, target_q_all)

        # Backprop through both critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = torch.zeros(1, device=self.device)
        if self.total_it % self.config.policy_delay == 0:
            # ---- 4) Actor update: improve policy a = π(s_{1:T}) ----
            # For sequence returns: maximize Q-values for ALL time steps in the sequence
            actor_actions_full, actor_hidden = self.actor(states_seq)
            actor_hidden = tuple(h.detach() for h in actor_hidden)
            if actor_actions_full.dim() == 2:
                actor_actions_full = actor_actions_full.unsqueeze(1)
            # Use the full policy-generated action sequence
            actor_actions_seq = actor_actions_full
            q1_actor, q2_actor = self.critic(states_seq, actor_actions_seq)
            # Use min(Q1, Q2) for actor update (TD3 standard: clipped double-Q for actor)
            # This prevents overestimation bias and makes the actor more conservative
            # For sequence returns: maximize Q-values across ALL time steps
            q_actor_min = torch.min(q1_actor.outputs, q2_actor.outputs)  # [B, T, 1]
            actor_loss = -q_actor_min.mean()  # Average over all time steps and batch

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # After updating the actor and critic, move targets a little bit
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)

        # Log Q-values at the FIRST time step of each sequence (closest to episode returns)
        # Also log the mean over all time steps for comparison
        q1_first_step = q1_current.outputs[:, 0, :].mean().item()  # [B, 1] -> scalar
        q2_first_step = q2_current.outputs[:, 0, :].mean().item()  # [B, 1] -> scalar
        q1_mean_all = q1_current.outputs.mean().item()  # Average over all [B, T, 1]
        q2_mean_all = q2_current.outputs.mean().item()  # Average over all [B, T, 1]
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": q1_first_step,  # Q-value at first time step (should match episode returns)
            "q2_mean": q2_first_step,  # Q-value at first time step (should match episode returns)
            "q1_mean_all": q1_mean_all,  # Average over all time steps (for comparison)
            "q2_mean_all": q2_mean_all,  # Average over all time steps (for comparison)
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        tau = self.config.tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)


__all__ = ["TD3Agent", "TD3Config"]



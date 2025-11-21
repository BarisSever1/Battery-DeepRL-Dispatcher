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
    """Hyperparameters controlling TD3 behaviour."""

    gamma: float = 0.99
    n_step: int = 6
    use_n_step: bool = True
    tau: float = 0.01
    policy_delay: int = 2
    lr: float = 7e-5
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    exploration_std: float = 0.5
    target_std: float = 0.5
    noise_clip: float = 0.4
    action_low: float = -1.0
    action_high: float = 1.0


class TD3Agent:
    """TD3 algorithm with LSTM actor/critics and action clamping."""

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
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or TD3Config()

        actor_kwargs = {
            "state_dim": state_dim,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_layers,
            "mlp_hidden_sizes": mlp_hidden_sizes,
        }
        critic_kwargs = {
            "state_dim": state_dim,
            "act_dim": action_dim,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_layers,
            "mlp_hidden_sizes": mlp_hidden_sizes,
        }

        self.actor = ActorLSTM(**actor_kwargs).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = CriticLSTM(**critic_kwargs).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        actor_lr = self.config.actor_lr or self.config.lr
        critic_lr = self.config.critic_lr or self.config.lr
        self.actor_optimizer: Optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer: Optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.total_it = 0

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(
        self,
        state_seq: np.ndarray,
        eval_mode: bool = False,
    ) -> np.ndarray:
        """Return an action for the given state (expects shape [B,T,F])."""

        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state_seq, dtype=torch.float32, device=self.device)
            action_tensor, _ = self.actor(state_tensor)

            if not eval_mode:
                noise = torch.randn_like(action_tensor) * float(self.config.exploration_std)
                action_tensor = (action_tensor + noise).clamp(self.config.action_low, self.config.action_high)

            action = action_tensor.cpu().numpy()

        return action

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, buffer: ReplayBuffer, batch_size: int, seq_len: int = 24) -> dict:
        self.total_it += 1

        batch = buffer.sample(batch_size, seq_len=seq_len)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards_raw = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones_raw = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        
        # Extract only the last timestep (t = seq_len-1) for rewards and dones
        # Handle both [B, T] and [B, T, 1] shapes
        if rewards_raw.dim() == 2:
            rewards_last = rewards_raw[:, -1:].clone()  # [B, 1]
        else:
            rewards_last = rewards_raw[:, -1, :].clone().unsqueeze(-1)  # [B, 1]
        
        if dones_raw.dim() == 2:
            dones_last = dones_raw[:, -1:].clone()  # [B, 1]
        else:
            dones_last = dones_raw[:, -1, :].clone().unsqueeze(-1)  # [B, 1]
        
        # Keep full sequences for states/actions (needed for LSTM)
        states_seq = states
        next_states_seq = next_states
        actions_seq = actions

        with torch.no_grad():
            # Optionally use n-step targets
            if self.config.use_n_step and "n_step_rewards" in batch:
                n_rewards_raw = torch.as_tensor(batch["n_step_rewards"], dtype=torch.float32, device=self.device)
                n_dones_raw = torch.as_tensor(batch["n_step_dones"], dtype=torch.float32, device=self.device)
                n_next_states_raw = torch.as_tensor(batch["n_step_next_states"], dtype=torch.float32, device=self.device)
                
                # Extract only the last timestep for n-step returns
                if n_rewards_raw.dim() == 2:
                    n_rewards_last = n_rewards_raw[:, -1:].clone()  # [B, 1]
                else:
                    n_rewards_last = n_rewards_raw[:, -1, :].clone().unsqueeze(-1)  # [B, 1]
                
                if n_dones_raw.dim() == 2:
                    n_dones_last = n_dones_raw[:, -1:].clone()  # [B, 1]
                else:
                    n_dones_last = n_dones_raw[:, -1, :].clone().unsqueeze(-1)  # [B, 1]
                
                # Use full sequence for next_states (needed for LSTM)
                n_next_states_seq = n_next_states_raw

                next_actions, hidden_next = self.actor_target(n_next_states_seq)
                # Detach hidden states to prevent gradient flow
                hidden_next = tuple(h.detach() for h in hidden_next)
                target_noise = torch.clamp(
                    torch.randn_like(next_actions) * self.config.target_std,
                    -self.config.noise_clip,
                    self.config.noise_clip,
                )
                next_actions = (next_actions + target_noise).clamp(
                    self.config.action_low,
                    self.config.action_high,
                )
                q1_next, q2_next = self.critic_target(n_next_states_seq, next_actions)
                # TD3 fix: use only last timestep Q-values
                min_q_next = torch.min(q1_next.outputs[:, -1, :], q2_next.outputs[:, -1, :])

                gamma_n = (self.config.gamma ** self.config.n_step)
                target_q = n_rewards_last + (1.0 - n_dones_last) * gamma_n * min_q_next
            else:
                next_actions, hidden_next = self.actor_target(next_states_seq)
                # Detach hidden states to prevent gradient flow
                hidden_next = tuple(h.detach() for h in hidden_next)
                target_noise = torch.clamp(
                    torch.randn_like(next_actions) * self.config.target_std,
                    -self.config.noise_clip,
                    self.config.noise_clip,
                )
                next_actions = (next_actions + target_noise).clamp(
                    self.config.action_low,
                    self.config.action_high,
                )

                q1_next, q2_next = self.critic_target(next_states_seq, next_actions)
                # TD3 fix: use only last timestep Q-values
                q1_next_values = q1_next.outputs[:, -1, :]
                q2_next_values = q2_next.outputs[:, -1, :]
                min_q_next = torch.min(q1_next_values, q2_next_values)
                target_q = rewards_last + (1.0 - dones_last) * self.config.gamma * min_q_next

        q1_current, q2_current = self.critic(states_seq, actions_seq)
        # TD3 fix: compute loss only on last timestep
        q1_last = q1_current.outputs[:, -1, :]
        q2_last = q2_current.outputs[:, -1, :]
        target_last = target_q
        q1_loss = nn.functional.mse_loss(q1_last, target_last)
        q2_loss = nn.functional.mse_loss(q2_last, target_last)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_it % self.config.policy_delay == 0:
            for param in self.critic.parameters():
                param.requires_grad = False

            actor_actions, hidden_actor = self.actor(states_seq)
            # Detach hidden states
            hidden_actor = tuple(h.detach() for h in hidden_actor)
            # TD3 fix: actor should optimize Q(s_t, π(s_t)) using last timestep only
            # actor_actions is already [B,1,1] from ActorLSTM fix
            q_actor, _ = self.critic(states_seq, actor_actions)
            # Use only last timestep Q-value
            actor_loss = -q_actor.outputs[:, -1, :].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": q1_current.outputs.mean().item(),
            "q2_mean": q2_current.outputs.mean().item(),
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



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
        """Return an action for the given state history (expects shape [B,T,F])."""

        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state_seq, dtype=torch.float32, device=self.device)
            action_seq, hidden = self.actor(state_tensor)
            hidden = tuple(h.detach() for h in hidden)

            if action_seq.dim() == 2:  # [B,A] -> [B,1,A]
                action_seq = action_seq.unsqueeze(1)

            actions_last = action_seq[:, -1:, :]  # [B,1,A]

            if not eval_mode:
                noise = torch.randn_like(actions_last) * float(self.config.exploration_std)
                noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
                actions_last = actions_last + noise

            actions_last = actions_last.clamp(self.config.action_low, self.config.action_high)
            actions_np = actions_last.squeeze(1).cpu().numpy()

        self.actor.train()
        return actions_np

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, buffer: ReplayBuffer, batch_size: int, seq_len: int = 24) -> dict:
        self.total_it += 1

        batch = buffer.sample(batch_size, seq_len=seq_len)
        states_seq = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions_seq = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards_raw = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states_seq = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones_raw = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        seq_len_actual = states_seq.shape[1]

        def _last_step(t: Tensor) -> Tensor:
            if t.dim() == 3:
                return t[:, -1, :].view(t.size(0), -1)
            if t.dim() == 2:
                return t[:, -1].view(t.size(0), 1)
            if t.dim() == 1:
                return t.view(t.size(0), 1)
            raise ValueError("Unexpected tensor rank")

        rewards_last = _last_step(rewards_raw)
        dones_last = _last_step(dones_raw)

        use_n_step = self.config.use_n_step and "n_step_rewards" in batch
        gamma_factor = self.config.gamma
        target_state_seq = next_states_seq

        if use_n_step:
            n_rewards_raw = torch.as_tensor(batch["n_step_rewards"], dtype=torch.float32, device=self.device)
            n_dones_raw = torch.as_tensor(batch["n_step_dones"], dtype=torch.float32, device=self.device)
            n_next_states_raw = torch.as_tensor(batch["n_step_next_states"], dtype=torch.float32, device=self.device)
            rewards_last = _last_step(n_rewards_raw)
            dones_last = _last_step(n_dones_raw)
            target_state_seq = n_next_states_raw
            gamma_factor = self.config.gamma ** self.config.n_step

        batch_size = states_seq.size(0)

        actions_last = actions_seq[:, -1:, :].clone()
        actions_broadcast = actions_last.expand(-1, seq_len_actual, -1)

        with torch.no_grad():
            next_actions_full, hidden_next = self.actor_target(target_state_seq)
            hidden_next = tuple(h.detach() for h in hidden_next)
            if next_actions_full.dim() == 2:
                next_actions_full = next_actions_full.unsqueeze(1)
            next_actions_last = next_actions_full[:, -1:, :]
            target_noise = torch.randn_like(next_actions_last) * self.config.target_std
            target_noise = target_noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_actions_last = next_actions_last + target_noise
            next_actions_last = next_actions_last.clamp(self.config.action_low, self.config.action_high)
            target_seq_len = target_state_seq.shape[1]
            next_actions_seq = next_actions_last.expand(-1, target_seq_len, -1)

            q1_next, q2_next = self.critic_target(target_state_seq, next_actions_seq)
            q1_next_hidden = tuple(h.detach() for h in q1_next.hidden)
            q2_next_hidden = tuple(h.detach() for h in q2_next.hidden)
            min_q_next = torch.min(q1_next.outputs[:, -1, :], q2_next.outputs[:, -1, :]).view(batch_size, 1)
            target_q = rewards_last + (1.0 - dones_last) * gamma_factor * min_q_next
            target_q = target_q.view(batch_size, 1)

        q1_current, q2_current = self.critic(states_seq, actions_broadcast)
        q1_last = q1_current.outputs[:, -1, :].view(batch_size, 1)
        q2_last = q2_current.outputs[:, -1, :].view(batch_size, 1)
        critic_loss = nn.functional.mse_loss(q1_last, target_q) + nn.functional.mse_loss(q2_last, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = torch.zeros(1, device=self.device)
        if self.total_it % self.config.policy_delay == 0:
            actor_actions_full, actor_hidden = self.actor(states_seq)
            actor_hidden = tuple(h.detach() for h in actor_hidden)
            if actor_actions_full.dim() == 2:
                actor_actions_full = actor_actions_full.unsqueeze(1)
            actor_last = actor_actions_full[:, -1:, :].view(batch_size, 1, -1)
            actor_actions_seq = actor_last.expand(-1, seq_len_actual, -1)
            q_actor, _ = self.critic(states_seq, actor_actions_seq)
            actor_loss = -q_actor.outputs[:, -1, :].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

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



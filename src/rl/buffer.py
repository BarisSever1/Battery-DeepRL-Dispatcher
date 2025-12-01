"""Replay buffer utilities for TD3 + LSTM training.

This module provides a numpy-based replay buffer that stores transitions
``(state, action, reward, next_state, done)`` and supports sampling either
single-step batches or contiguous sequences of length ``seq_len``. A small
``FrameStacker`` helper is included to retrieve the most recent ``k`` frames for
diagnostics or custom batching logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


ArrayLike = np.ndarray


class ReplayBuffer:
    """
    Fixed-size replay buffer backed by pre-allocated numpy arrays.

    In plain terms: this is a big circular list of past experiences
    (state, action, reward, next_state, done). The agent stores every
    step here and later samples random mini-batches for training. This
    is standard practice in DDPG/TD3 and helps break temporal
    correlations in the data.

    Args:
        state_dim: Dimensionality of the flattened observation vector.
        action_dim: Dimensionality of the flattened action vector.
        capacity: Maximum number of transitions to store.
        dtype: Numpy dtype used for continuous values (default ``np.float32``).
        n_step: Horizon for n-step return (default 6).
        gamma: Discount factor used for n-step computation (default 1.0).

    Notes:
        * Transitions are stored in a circular buffer. Once capacity is reached,
          the oldest entries are overwritten.
        * ``sample`` can return either single steps (``seq_len=1``) or sequences
          of contiguous steps that do not cross episode boundaries.
        * When ``n_step >= 2``, the sampled batch additionally includes
          ``n_step_rewards``, ``n_step_next_states``, and ``n_step_dones``
          computed within each returned sequence using the provided ``gamma``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int = 200_000,
        dtype: np.dtype = np.float32,
        n_step: int = 6,
        gamma: float = 1.0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.dtype = dtype
        self.n_step = int(max(1, n_step))
        self.gamma = float(gamma)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=self.dtype)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=self.dtype)
        self.rewards = np.zeros((self.capacity,), dtype=self.dtype)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=self.dtype)
        self.dones = np.zeros((self.capacity,), dtype=self.dtype)

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: ArrayLike,
        action: ArrayLike,
        reward: float,
        next_state: ArrayLike,
        done: bool,
    ) -> None:
        """
        Insert a transition into the buffer.

        Args:
            state:      observation before taking the action.
            action:     action taken by the agent.
            reward:     scalar reward received after the step.
            next_state: observation after the environment step.
            done:       True if the episode ended at this step.
        """
        idx = self.position
        self.states[idx] = np.asarray(state, dtype=self.dtype)
        self.actions[idx] = np.asarray(action, dtype=self.dtype)
        self.rewards[idx] = np.asarray(reward, dtype=self.dtype)
        self.next_states[idx] = np.asarray(next_state, dtype=self.dtype)
        self.dones[idx] = np.asarray(done, dtype=self.dtype)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int = 1) -> Dict[str, ArrayLike]:
        """
        Sample a batch of transitions or contiguous sequences.

        Args:
            batch_size: Number of samples to draw.
            seq_len: Length of each sampled sequence (defaults to ``1``).

        Returns:
            Dictionary containing numpy arrays for ``states``, ``actions``,
            ``rewards``, ``next_states``, and ``dones``. For ``seq_len > 1`` the
            arrays have shape ``[batch_size, seq_len, *feature_dims]``. Rewards
            and dones retain a trailing dimension of length ``seq_len``.

            If ``self.n_step >= 2``, also returns:
            - ``n_step_rewards`` with shape ``[batch_size, seq_len]``
            - ``n_step_next_states`` with shape ``[batch_size, seq_len, state_dim]``
            - ``n_step_dones`` with shape ``[batch_size, seq_len]`` indicating
               whether a terminal was encountered before reaching ``n_step``
        """

        if self.size == 0:
            # No data to train on yet
            raise ValueError("Cannot sample from an empty replay buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if seq_len > self.size:
            raise ValueError("seq_len cannot exceed the number of stored items")

        # 1) choose buffer indices so that we do not cross episode boundaries
        indices = self._sample_indices(batch_size, seq_len)
        # 2) gather contiguous sequences ending at each chosen index
        states = self._gather_sequences(self.states, indices, seq_len)
        actions = self._gather_sequences(self.actions, indices, seq_len)
        rewards = self._gather_sequences(self.rewards[:, None], indices, seq_len).squeeze(-1)
        next_states = self._gather_sequences(self.next_states, indices, seq_len)
        dones = self._gather_sequences(self.dones[:, None], indices, seq_len).squeeze(-1)

        batch = {
            "states": states.astype(self.dtype, copy=False),
            "actions": actions.astype(self.dtype, copy=False),
            "rewards": rewards.astype(self.dtype, copy=False),
            "next_states": next_states.astype(self.dtype, copy=False),
            "dones": dones.astype(self.dtype, copy=False),
            "indices": indices,
        }

        # Compute n-step targets within each sampled sequence, without crossing episode boundaries
        if self.n_step >= 2:
            B, T = rewards.shape
            n_step_rewards = np.zeros((B, T), dtype=self.dtype)
            n_step_dones = np.zeros((B, T), dtype=self.dtype)
            n_step_next_states = np.zeros((B, T, self.state_dim), dtype=self.dtype)

            gamma = self.gamma
            n = self.n_step

            for b in range(B):
                base_buf_index = indices[b]
                for t in range(T):
                    G = 0.0
                    g = 1.0
                    done_flag = 0.0

                    start_idx = (base_buf_index - (T - 1 - t)) % self.capacity
                    final_idx = start_idx

                    for k in range(n):
                        idx_k = (start_idx + k) % self.capacity
                        G += g * self.rewards[idx_k]
                        final_idx = idx_k
                        if self.dones[idx_k] > 0.5:
                            done_flag = 1.0
                            break
                        g *= gamma

                    n_step_rewards[b, t] = G
                    n_step_dones[b, t] = done_flag
                    n_step_next_states[b, t] = self.next_states[final_idx]

            batch["n_step_rewards"] = n_step_rewards
            batch["n_step_dones"] = n_step_dones
            batch["n_step_next_states"] = n_step_next_states

        return batch

    def clear(self) -> None:
        """Empty the buffer while keeping allocated memory."""

        self.position = 0
        self.size = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_indices(self, batch_size: int, seq_len: int) -> np.ndarray:
        """
        Pick valid end-indices for sequences of length ``seq_len``.

        We make sure that the chosen indices do not "jump over" episode
        boundaries (where ``done`` is True), so each sampled sequence
        belongs to a single episode.
        """
        indices = np.empty(batch_size, dtype=np.int64)
        attempts = 0
        max_attempts = batch_size * 50

        for i in range(batch_size):
            while True:
                if attempts >= max_attempts:
                    raise RuntimeError("Unable to sample a valid batch without crossing episode boundaries")
                if self.size == self.capacity:
                    idx = np.random.randint(0, self.size)
                else:
                    idx = np.random.randint(seq_len - 1, self.size)

                attempts += 1
                if self._is_valid_index(idx, seq_len):
                    indices[i] = idx
                    break

        return indices

    def _is_valid_index(self, idx: int, seq_len: int) -> bool:
        if seq_len == 1:
            return True

        if self.size < self.capacity and idx - (seq_len - 1) < 0:
            return False

        for offset in range(1, seq_len):
            prev_idx = (idx - offset) % self.capacity
            if self.size < self.capacity and prev_idx >= self.size:
                return False
            if self.dones[prev_idx] > 0.5:
                return False
        return True

    def _gather_sequences(
        self,
        array: ArrayLike,
        indices: Iterable[int],
        seq_len: int,
    ) -> ArrayLike:
        """
        Given a 1D buffer ``array`` and some end indices, build a tensor of
        shape ``[batch_size, seq_len, ...]`` where each row is a contiguous
        slice ending at the corresponding index.
        """
        array = np.asarray(array)
        batch_size = len(indices)
        suffix = array.shape[1:]
        out = np.empty((batch_size, seq_len, *suffix), dtype=array.dtype)

        for i, idx in enumerate(indices):
            for step in range(seq_len):
                arr_idx = (idx - (seq_len - 1 - step)) % self.capacity
                out[i, step] = array[arr_idx]

        return out


@dataclass
class FrameStacker:
    """Utility to fetch the last ``k`` observations from the buffer.

    Args:
        buffer: Source replay buffer.
        k: Number of frames to stack (``>= 1``).

    Notes:
        Raises ``ValueError`` if insufficient history exists (e.g., requesting 4
        frames when the sampled index is less than 3 and the buffer has not yet
        wrapped).
    """

    buffer: ReplayBuffer
    k: int = 1

    def stack(self, idx: int) -> np.ndarray:
        if self.k <= 0:
            raise ValueError("k must be at least 1")
        if idx < 0 or idx >= len(self.buffer):
            raise IndexError("Index out of range for the replay buffer")

        if self.buffer.size < self.buffer.capacity and idx - (self.k - 1) < 0:
            raise ValueError("Not enough history to stack the requested number of frames")

        indices = np.array([idx], dtype=np.int64)
        stacked = self.buffer._gather_sequences(self.buffer.states, indices, self.k)
        return stacked[0]


__all__ = ["ReplayBuffer", "FrameStacker"]



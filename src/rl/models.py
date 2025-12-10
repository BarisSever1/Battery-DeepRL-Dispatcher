"""Neural network modules for TD3 + LSTM agent.

This module defines the actor and twin critic networks used by the TD3
algorithm. Both modules share an LSTM backbone (hidden size 64) followed by a
two-layer multilayer perceptron (MLP) with ReLU activations.     The actor maps
normalized environment states to actions shaped ``[batch, time, 1]`` in the
``[0, 1]`` interval (interpreted based on peak windows: discharge in peaks, charge outside), while the
critics return Q-value sequences of shape ``[batch, time, 1]`` for
state-action pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


def _build_mlp(input_dim: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    """Construct a simple feedforward network with ReLU activations."""

    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    return nn.Sequential(*layers)


@dataclass
class LSTMOutput:
    """
    Small helper class to bundle what an LSTM block returns:
      * outputs: the sequence of outputs for every time step.
      * hidden: the hidden state tuple (h_n, c_n) for maintaining state across sequences.
    """

    outputs: Tensor
    hidden: Tuple[Tensor, Tensor] = None


class ActorLSTM(nn.Module):
    """LSTM-based actor that outputs continuous actions in ``[0, 1]``.

    Args:
        state_dim: Number of input features per time step.
        lstm_hidden_size: Hidden size of the LSTM backbone.
        lstm_num_layers: Number of stacked LSTM layers.
        mlp_hidden_sizes: Sizes of the fully-connected layers after the LSTM.

    Notes:
        The forward method accepts state sequences shaped ``[batch, time, F]``
        and returns an action sequence ``[batch, time, 1]`` (bounded to
        ``[0, 1]``). The LSTM processes the full sequence internally with
        hidden states reset at the start of each sequence.
    """

    def __init__(
        self,
        state_dim: int,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        mlp_hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        # Recurrent part that looks at the sequence over time
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        # Feed-forward layers applied to every time step of the LSTM output
        self.backbone = _build_mlp(lstm_hidden_size, mlp_hidden_sizes)
        self.head = nn.Linear(mlp_hidden_sizes[-1] if mlp_hidden_sizes else lstm_hidden_size, 1)

    def forward(
        self,
        state_seq: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through the actor network.
        
        Args:
            state_seq: State sequence of shape [batch, time, features]
            hidden: Optional hidden state tuple (h_n, c_n) from previous forward pass.
                   If None, hidden state is initialized to zeros.
        
        Returns:
            actions: Action sequence of shape [batch, time, 1]
            hidden: Hidden state tuple (h_n, c_n) for maintaining state across sequences
        """
        if state_seq.dim() != 3:
            raise ValueError("state_seq must have shape [batch, time, features]")

        # Run the sequence through the LSTM with optional hidden state
        lstm_out, hidden_out = self.lstm(state_seq, hidden)
        # Apply MLP to each time step
        features = self.backbone(lstm_out)
        # Map to a single action value per time step, then squash to [0, 1]
        raw_action = self.head(features)
        actions = torch.sigmoid(raw_action)  # shape [B,T,1], range [0, 1]
        return actions, hidden_out


class _CriticHead(nn.Module):
    """
    Single critic head: LSTM backbone + MLP predicting scalar Q-values.

    Given a sequence of (state, action) pairs it produces Q-values for ALL time steps.
    This enables sequence returns learning where we learn from all time steps in each sequence.
    """

    def __init__(
        self,
        input_dim: int,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        mlp_hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        # Sequence encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.backbone = _build_mlp(lstm_hidden_size, mlp_hidden_sizes)
        last_dim = mlp_hidden_sizes[-1] if mlp_hidden_sizes else lstm_hidden_size
        self.head = nn.Linear(last_dim, 1)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> LSTMOutput:
        """
        Forward pass through the critic head.
        
        Args:
            x: Input sequence of shape [batch, time, features]
            hidden: Optional hidden state tuple (h_n, c_n) from previous forward pass.
                   If None, hidden state is initialized to zeros.
        
        Returns:
            LSTMOutput with outputs and hidden state
        """
        if x.dim() != 3:
            raise ValueError("Input must have shape [batch, time, features]")

        # Encode the sequence with optional hidden state
        lstm_out, hidden_out = self.lstm(x, hidden)
        features = self.backbone(lstm_out)
        # Return Q-values for ALL time steps (shape [B,T,1]) for sequence returns
        q_values = self.head(features)  # [B, T, 1]
        return LSTMOutput(outputs=q_values, hidden=hidden_out)


class CriticLSTM(nn.Module):
    """
    Twin-critic module.

    Wraps two independent `_CriticHead`s (q1 and q2) so TD3 can use
    clipped double-Q: we take the minimum of the two critics to reduce
    over-estimation bias.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int = 1,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        mlp_hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        # Critic sees state and action together at each time step
        input_dim = state_dim + act_dim
        self.q1 = _CriticHead(
            input_dim=input_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            mlp_hidden_sizes=mlp_hidden_sizes,
        )
        self.q2 = _CriticHead(
            input_dim=input_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            mlp_hidden_sizes=mlp_hidden_sizes,
        )

    def forward(
        self,
        state_seq: Tensor,
        action_seq: Tensor,
        hidden: Optional[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[LSTMOutput, LSTMOutput]:
        """
        Forward pass through the twin critics.
        
        Args:
            state_seq: State sequence of shape [batch, time, state_features]
            action_seq: Action sequence of shape [batch, time, action_features] or [batch, 1, action_features]
            hidden: Optional tuple of hidden states (q1_hidden, q2_hidden) from previous forward pass.
                   Each is a tuple (h_n, c_n). If None, hidden states are initialized to zeros.
        
        Returns:
            Tuple of (q1_output, q2_output) where each is an LSTMOutput with outputs and hidden state
        """
        # Expect [B,T,F] for states and [B,T,A] (or [B,1,A]) for actions.
        if state_seq.dim() != 3:
            raise ValueError("state_seq must have shape [batch, time, state_features]")
        if action_seq.dim() == 2:
            # Normalize [B,A] -> [B,1,A]
            action_seq = action_seq.unsqueeze(1)
        if action_seq.dim() != 3:
            raise ValueError("action_seq must have shape [batch, time, action_features]")

        B, T_s, _ = state_seq.shape
        B_a, T_a, _ = action_seq.shape
        if B != B_a:
            raise ValueError(f"Batch size mismatch: states B={B}, actions B={B_a}")

        # If actions have a single time step but states have multiple, repeat across time.
        if T_a == 1 and T_s > 1:
            action_seq = action_seq.repeat(1, T_s, 1)
            T_a = T_s

        if T_s != T_a:
            raise ValueError(f"Time dimension mismatch: states T={T_s}, actions T={T_a}")

        critic_input = torch.cat([state_seq, action_seq], dim=-1)  # [B,T,F+A]
        
        # Extract hidden states for q1 and q2
        q1_hidden = hidden[0] if hidden is not None else None
        q2_hidden = hidden[1] if hidden is not None else None
        
        q1 = self.q1(critic_input, q1_hidden)
        q2 = self.q2(critic_input, q2_hidden)
        return q1, q2


__all__ = ["ActorLSTM", "CriticLSTM", "LSTMOutput"]



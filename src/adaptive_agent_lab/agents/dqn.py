"""Masked Deep Q-Network agent backed by the project's NumPy MLP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from adaptive_agent_lab.agents.base import Agent
from adaptive_agent_lab.agents.planning import guided_action, navigation_progress
from adaptive_agent_lab.environment.contracts import Action, Transition, WarehouseSnapshot
from adaptive_agent_lab.environment.observation import ACTIONS, ObservationEncoder, ObservationSpec
from adaptive_agent_lab.learning.network import MLPQNetwork
from adaptive_agent_lab.learning.replay import ReplayBuffer

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DQNConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 32
    replay_capacity: int = 10_000
    warmup_steps: int = 200
    update_every: int = 1
    target_sync_interval: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    max_grad_norm: float = 10.0
    guidance_probability: float = 0.0
    guidance_bias: float = 0.0
    progress_reward: float = 0.0

    def __post_init__(self) -> None:
        if len(self.hidden_sizes) not in (1, 2) or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain one or two positive widths")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("learning rate and gradient norm must be positive")
        for name in (
            "batch_size",
            "replay_capacity",
            "update_every",
            "target_sync_interval",
            "epsilon_decay_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0.0 <= self.guidance_probability <= 1.0:
            raise ValueError("guidance_probability must be in [0, 1]")
        if not np.isfinite(self.guidance_bias) or self.guidance_bias < 0.0:
            raise ValueError("guidance_bias must be finite and non-negative")
        if not np.isfinite(self.progress_reward) or self.progress_reward < 0.0:
            raise ValueError("progress_reward must be finite and non-negative")


class DQNAgent(Agent):
    """DQN with invalid-action masking and a deterministic replay stream."""

    name = "dqn"
    trainable = True

    def __init__(self, config: DQNConfig | None = None) -> None:
        super().__init__()
        config = DQNConfig() if config is None else config
        if not isinstance(config, DQNConfig):
            raise TypeError("config must be a DQNConfig")
        self.config = config
        self._encoder: ObservationEncoder | None = None
        self._template: WarehouseSnapshot | None = None
        self._network: MLPQNetwork | None = None
        self._target: MLPQNetwork | None = None
        self._replay: ReplayBuffer[npt.NDArray[np.float64], int] | None = None
        self._environment_steps = 0
        self._updates = 0
        self._last_vector: FloatArray | None = None
        self._last_mask: npt.NDArray[np.bool_] | None = None
        self._last_action_index: int | None = None
        self._last_loss: float | None = None

    @property
    def epsilon(self) -> float:
        fraction = min(self._environment_steps / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    @property
    def replay_size(self) -> int:
        return 0 if self._replay is None else len(self._replay)

    @property
    def update_count(self) -> int:
        return self._updates

    @property
    def last_loss(self) -> float | None:
        return self._last_loss

    def _reset(self, snapshot: WarehouseSnapshot) -> None:
        spec = ObservationSpec.from_snapshot(snapshot)
        encoder = ObservationEncoder(spec)
        if self._network is None:
            network_seed = int(self._rng.integers(0, 2**32 - 1))
            replay_seed = int(self._rng.integers(0, 2**32 - 1))
            self._network = MLPQNetwork(
                spec.vector_size,
                len(ACTIONS),
                self.config.hidden_sizes,
                seed=network_seed,
            )
            self._target = self._network.copy()
            self._replay = ReplayBuffer(self.config.replay_capacity, seed=replay_seed)
        elif self._network.input_dim != spec.vector_size:
            raise ValueError("DQN cannot retain one network across incompatible observation specs")
        self._encoder = encoder
        self._template = snapshot
        self._last_vector = None
        self._last_mask = None
        self._last_action_index = None

    def _act(self, snapshot: WarehouseSnapshot, *, explore: bool) -> Action:
        encoder, network = self._ready()
        vector = encoder.vector(snapshot).astype(np.float64)
        mask = encoder.action_mask(snapshot)
        valid_indices = np.flatnonzero(mask)
        if valid_indices.size == 0:
            action_index = ACTIONS.index(Action.WAIT)
        elif explore and self._rng.random() < self.config.guidance_probability:
            action_index = ACTIONS.index(guided_action(snapshot))
        elif explore and self._rng.random() < self.epsilon:
            action_index = int(self._rng.choice(valid_indices))
        else:
            values = network.predict(vector)
            masked = np.where(mask, values, -np.inf)
            if self.config.guidance_bias:
                guide_index = ACTIONS.index(guided_action(snapshot))
                if mask[guide_index]:
                    masked[guide_index] += self.config.guidance_bias
            action_index = int(np.argmax(masked))
        self._last_vector = vector
        self._last_mask = mask.copy()
        self._last_action_index = action_index
        return ACTIONS[action_index]

    def observe(self, transition: Transition) -> None:
        encoder, _ = self._ready()
        if self._last_vector is None or self._last_action_index is None:
            raise RuntimeError("act must be called before observe")
        if ACTIONS[self._last_action_index] is not transition.action:
            raise ValueError("observed transition action does not match the selected action")
        next_snapshot = self._snapshot_with_state(transition.next_state)
        next_vector = encoder.vector(next_snapshot).astype(np.float64)
        next_mask = encoder.action_mask(next_snapshot)
        reward = transition.reward
        if transition.action.is_movement and self.config.progress_reward:
            assert self._template is not None
            current_snapshot = self._snapshot_with_state(transition.state)
            reward += self.config.progress_reward * navigation_progress(
                current_snapshot, next_snapshot
            )
        if self.learning_enabled:
            replay = self._require_replay()
            replay.add(
                self._pack(self._last_vector, self._last_mask),
                self._last_action_index,
                reward,
                self._pack(next_vector, next_mask),
                transition.terminated,
            )
            self._environment_steps += 1
            if self._should_update():
                self._learn_batch()
        self._last_vector = None
        self._last_mask = None
        self._last_action_index = None

    def q_values(self, snapshot: WarehouseSnapshot) -> FloatArray:
        encoder, network = self._ready()
        return network.predict(encoder.vector(snapshot).astype(np.float64))

    def state_dict(self) -> dict[str, object]:
        _, network = self._ready()
        assert self._target is not None
        return {
            "format_version": 1,
            "agent": self.name,
            "config": {**asdict(self.config), "hidden_sizes": list(self.config.hidden_sizes)},
            "environment_steps": self._environment_steps,
            "updates": self._updates,
            "network": network.state_dict(),
            "target": self._target.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("format_version") != 1 or state.get("agent") != self.name:
            raise ValueError("unsupported DQN agent state")
        raw_network = state.get("network")
        raw_target = state.get("target")
        if not isinstance(raw_network, Mapping) or not isinstance(raw_target, Mapping):
            raise ValueError("DQN state must contain network and target mappings")
        network = MLPQNetwork.from_state_dict(raw_network)
        target = MLPQNetwork.from_state_dict(raw_target)
        if network.layer_sizes != target.layer_sizes:
            raise ValueError("online and target network architectures must match")
        self._network = network
        self._target = target
        self._environment_steps = _state_nonnegative_int(state, "environment_steps")
        self._updates = _state_nonnegative_int(state, "updates")

    def clear_learning(self) -> None:
        self._network = None
        self._target = None
        self._replay = None
        self._environment_steps = 0
        self._updates = 0
        self._last_loss = None

    def _should_update(self) -> bool:
        replay = self._require_replay()
        minimum = max(self.config.batch_size, self.config.warmup_steps)
        return len(replay) >= minimum and self._environment_steps % self.config.update_every == 0

    def _learn_batch(self) -> None:
        network = self._require_network()
        assert self._target is not None
        batch = self._require_replay().sample_batch(self.config.batch_size)
        input_dim = network.input_dim
        states = batch.states[:, :input_dim]
        next_states = batch.next_states[:, :input_dim]
        next_masks = batch.next_states[:, input_dim:] > 0.5
        next_values = self._target.predict(next_states)
        masked_values = np.where(next_masks, next_values, -np.inf)
        has_action = np.any(next_masks, axis=1)
        bootstrap = np.zeros(batch.states.shape[0], dtype=np.float64)
        bootstrap[has_action] = np.max(masked_values[has_action], axis=1)
        targets = batch.rewards + self.config.gamma * (~batch.dones) * bootstrap
        self._last_loss = network.train_td_batch(
            states,
            batch.actions,
            targets,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
        )
        self._updates += 1
        self._record_learning_update()
        if self._updates % self.config.target_sync_interval == 0:
            self._target = network.copy()

    def _pack(
        self,
        vector: FloatArray,
        mask: npt.NDArray[np.bool_] | None,
    ) -> FloatArray:
        if mask is None:
            raise RuntimeError("an action mask is required for replay")
        return np.concatenate((vector, mask.astype(np.float64)))

    def _snapshot_with_state(self, state: object) -> WarehouseSnapshot:
        from adaptive_agent_lab.environment.contracts import WarehouseState

        if not isinstance(state, WarehouseState):
            raise TypeError("state must be a WarehouseState")
        if self._template is None:
            raise RuntimeError("agent has not been reset")
        return WarehouseSnapshot(
            warehouse_map=self._template.map,
            state=state,
            orders=self._template.orders,
            horizon=self._template.horizon,
            battery_capacity=self._template.battery_capacity,
        )

    def _ready(self) -> tuple[ObservationEncoder, MLPQNetwork]:
        if self._encoder is None:
            raise RuntimeError("agent must be reset before use")
        return self._encoder, self._require_network()

    def _require_network(self) -> MLPQNetwork:
        if self._network is None:
            raise RuntimeError("DQN network is not initialized")
        return self._network

    def _require_replay(self) -> ReplayBuffer[npt.NDArray[np.float64], int]:
        if self._replay is None:
            raise RuntimeError("DQN replay buffer is not initialized")
        return self._replay


def _state_nonnegative_int(state: Mapping[str, object], key: str) -> int:
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def config_from_mapping(data: Mapping[str, object]) -> DQNConfig:
    """Construct a validated config from JSON-compatible input."""

    hidden = data.get("hidden_sizes", (64, 64))
    if not isinstance(hidden, Sequence) or isinstance(hidden, (str, bytes)):
        raise TypeError("hidden_sizes must be an array")
    return DQNConfig(
        hidden_sizes=tuple(int(value) for value in hidden),
        gamma=float(cast(str | int | float, data.get("gamma", 0.99))),
        learning_rate=float(cast(str | int | float, data.get("learning_rate", 1e-3))),
        batch_size=int(cast(str | int | float, data.get("batch_size", 32))),
        replay_capacity=int(
            cast(str | int | float, data.get("replay_capacity", 10_000))
        ),
        warmup_steps=int(cast(str | int | float, data.get("warmup_steps", 200))),
        update_every=int(cast(str | int | float, data.get("update_every", 1))),
        target_sync_interval=int(
            cast(str | int | float, data.get("target_sync_interval", 200))
        ),
        epsilon_start=float(cast(str | int | float, data.get("epsilon_start", 1.0))),
        epsilon_end=float(cast(str | int | float, data.get("epsilon_end", 0.05))),
        epsilon_decay_steps=int(
            cast(str | int | float, data.get("epsilon_decay_steps", 5_000))
        ),
        max_grad_norm=float(cast(str | int | float, data.get("max_grad_norm", 10.0))),
        guidance_probability=float(
            cast(str | int | float, data.get("guidance_probability", 0.0))
        ),
        guidance_bias=float(cast(str | int | float, data.get("guidance_bias", 0.0))),
        progress_reward=float(
            cast(str | int | float, data.get("progress_reward", 0.0))
        ),
    )


__all__ = ["DQNAgent", "DQNConfig", "config_from_mapping"]

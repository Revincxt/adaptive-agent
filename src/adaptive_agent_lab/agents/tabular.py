"""Masked tabular Q-learning and deterministic Dyna-Q agents."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np

from adaptive_agent_lab.agents.base import Agent
from adaptive_agent_lab.agents.planning import guided_action, navigation_progress
from adaptive_agent_lab.environment.contracts import (
    Action,
    Order,
    Transition,
    WarehouseMap,
    WarehouseSnapshot,
    WarehouseState,
)
from adaptive_agent_lab.environment.observation import (
    ACTION_INDEX,
    ACTIONS,
    ObservationEncoder,
    ObservationSpec,
    TabularState,
)

TABULAR_AGENT_SCHEMA_VERSION = 1
QValues: TypeAlias = tuple[float, ...]
JsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class _Experience:
    state: TabularState
    action_index: int
    reward: float
    next_state: TabularState
    terminated: bool
    next_actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ModelEntry:
    reward: float
    next_state: TabularState
    terminated: bool
    next_actions: tuple[int, ...]


class QLearningAgent(Agent):
    """Masked one-step Q-learning over :class:`ObservationEncoder` states.

    Learning state deliberately survives :meth:`Agent.reset`; reset denotes a
    new episode, not a new experiment.  Call :meth:`clear` explicitly to remove
    learned values and restart the epsilon schedule.
    """

    name = "q-learning"
    trainable = True

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        guidance_probability: float = 0.0,
        progress_reward: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = _probability("alpha", alpha, positive=True)
        self.gamma = _probability("gamma", gamma)
        self.epsilon_initial = _probability("epsilon", epsilon)
        self.epsilon_decay = _probability(
            "epsilon_decay", epsilon_decay, positive=True
        )
        self.epsilon_min = _probability("epsilon_min", epsilon_min)
        self.guidance_probability = _probability(
            "guidance_probability", guidance_probability
        )
        if not math.isfinite(progress_reward) or progress_reward < 0.0:
            raise ValueError("progress_reward must be finite and non-negative")
        self.progress_reward = float(progress_reward)
        if self.epsilon_min > self.epsilon_initial:
            raise ValueError("epsilon_min cannot exceed epsilon")

        self._epsilon = self.epsilon_initial
        self._q_table: dict[TabularState, list[float]] = {}
        self._encoder: ObservationEncoder | None = None
        self._warehouse_map: WarehouseMap | None = None
        self._orders: tuple[Order, ...] = ()
        self._horizon: int | None = None
        self._battery_capacity: int | None = None
        self._pending_snapshot: WarehouseSnapshot | None = None
        self._pending_action: Action | None = None

    @property
    def epsilon(self) -> float:
        """Current exploration probability after decay."""

        return self._epsilon

    @property
    def epsilon_current(self) -> float:
        """Explicit alias useful in metric and configuration reports."""

        return self._epsilon

    @property
    def q_table(self) -> Mapping[TabularState, QValues]:
        """Return an immutable defensive view of learned action values."""

        values = {
            state: tuple(action_values)
            for state, action_values in sorted(self._q_table.items())
        }
        return MappingProxyType(values)

    def q_values(self, state: TabularState) -> QValues:
        """Return all action values without inserting an unseen state."""

        normalized = _runtime_state(state, "state")
        values = self._q_table.get(normalized)
        if values is None:
            return (0.0,) * len(ACTIONS)
        return tuple(values)

    def q_value(self, state: TabularState, action: Action) -> float:
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        return self.q_values(state)[ACTION_INDEX[action]]

    def set_q_value(self, state: TabularState, action: Action, value: float) -> None:
        """Set one value, primarily for controlled experiments and fixtures."""

        normalized = _runtime_state(state, "state")
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        numeric = _finite_float(value, "value")
        self._values_for(normalized)[ACTION_INDEX[action]] = numeric

    def clear(self) -> None:
        """Discard learned values and restore the initial epsilon."""

        self._q_table.clear()
        self._epsilon = self.epsilon_initial
        self._pending_snapshot = None
        self._pending_action = None

    def _reset(self, snapshot: WarehouseSnapshot) -> None:
        self._encoder = ObservationEncoder(ObservationSpec.from_snapshot(snapshot))
        self._warehouse_map = snapshot.map
        self._orders = snapshot.orders
        self._horizon = snapshot.horizon
        self._battery_capacity = snapshot.battery_capacity
        self._pending_snapshot = None
        self._pending_action = None

    def _act(self, snapshot: WarehouseSnapshot, *, explore: bool) -> Action:
        encoder = self._require_encoder()
        state = encoder.tabular(snapshot)
        mask = encoder.action_mask(snapshot)
        feasible = tuple(int(index) for index in np.flatnonzero(mask))
        if not feasible:
            action = Action.WAIT
        elif explore and self._rng.random() < self.guidance_probability:
            action = guided_action(snapshot)
        elif explore and self._rng.random() < self._epsilon:
            action = ACTIONS[int(self._rng.choice(feasible))]
        else:
            values = self._q_table.get(state)
            action_index = min(
                feasible,
                key=lambda index: (-(values[index] if values is not None else 0.0), index),
            )
            action = ACTIONS[action_index]
        self._pending_snapshot = snapshot
        self._pending_action = action
        return action

    def observe(self, transition: Transition) -> None:
        experience = self._experience_from_transition(transition)
        if self.learning_enabled:
            self._q_update(experience)
            self._record_learning_update()
            self._after_real_experience(experience)
            self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
        self._pending_snapshot = None
        self._pending_action = None

    def _after_real_experience(self, experience: _Experience) -> None:
        del experience

    def _experience_from_transition(self, transition: Transition) -> _Experience:
        if not isinstance(transition, Transition):
            raise TypeError("transition must be an immutable Transition")
        cached_snapshot = self._pending_snapshot
        cached_action = self._pending_action
        if cached_snapshot is None or cached_action is None:
            raise RuntimeError("observe requires a preceding act call")
        if transition.state != cached_snapshot.state:
            raise ValueError("transition state does not match the state passed to act")
        if transition.action is not cached_action:
            raise ValueError("transition action does not match the action returned by act")

        encoder = self._require_encoder()
        next_snapshot = self._snapshot_for(transition.next_state)
        next_mask = encoder.action_mask(next_snapshot)
        reward = transition.reward
        if cached_action.is_movement and self.progress_reward:
            reward += self.progress_reward * navigation_progress(
                cached_snapshot, next_snapshot
            )
        return _Experience(
            state=encoder.tabular(cached_snapshot),
            action_index=ACTION_INDEX[cached_action],
            reward=reward,
            next_state=encoder.tabular(next_snapshot),
            terminated=transition.terminated,
            next_actions=tuple(int(index) for index in np.flatnonzero(next_mask)),
        )

    def _snapshot_for(self, state: WarehouseState) -> WarehouseSnapshot:
        if self._warehouse_map is None or self._horizon is None:
            raise RuntimeError("agent must be reset before observing transitions")
        if self._battery_capacity is None:
            raise RuntimeError("agent observation context is incomplete")
        return WarehouseSnapshot(
            warehouse_map=self._warehouse_map,
            state=state,
            orders=self._orders,
            horizon=self._horizon,
            battery_capacity=self._battery_capacity,
        )

    def _q_update(self, experience: _Experience) -> None:
        values = self._values_for(experience.state)
        current = values[experience.action_index]
        bootstrap = 0.0
        if not experience.terminated and experience.next_actions:
            next_values = self._q_table.get(experience.next_state)
            if next_values is not None:
                bootstrap = max(next_values[index] for index in experience.next_actions)
        target = experience.reward + self.gamma * bootstrap
        values[experience.action_index] = current + self.alpha * (target - current)

    def _values_for(self, state: TabularState) -> list[float]:
        return self._q_table.setdefault(state, [0.0] * len(ACTIONS))

    def _require_encoder(self) -> ObservationEncoder:
        if self._encoder is None:
            raise RuntimeError("agent must be reset before acting or observing")
        return self._encoder

    def _hyperparameters(self) -> dict[str, JsonScalar]:
        return {
            "alpha": self.alpha,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_initial": self.epsilon_initial,
            "epsilon_min": self.epsilon_min,
            "gamma": self.gamma,
            "guidance_probability": self.guidance_probability,
            "progress_reward": self.progress_reward,
        }

    def state_dict(self) -> dict[str, object]:
        """Return deterministic state composed solely of JSON values."""

        q_rows: list[dict[str, object]] = []
        for state, values in sorted(self._q_table.items()):
            q_rows.append({"state": list(state), "values": list(values)})
        return {
            "agent": self.name,
            "epsilon": self._epsilon,
            "hyperparameters": self._hyperparameters(),
            "q_table": q_rows,
            "schema_version": TABULAR_AGENT_SCHEMA_VERSION,
        }

    def load_state_dict(self, data: Mapping[str, object]) -> None:
        """Load values transactionally after strict schema validation."""

        q_table, epsilon = self._decode_base_state(data)
        self._q_table = q_table
        self._epsilon = epsilon
        self._pending_snapshot = None
        self._pending_action = None

    def _decode_base_state(
        self,
        data: Mapping[str, object],
        *,
        extra_keys: frozenset[str] = frozenset(),
    ) -> tuple[dict[TabularState, list[float]], float]:
        if not isinstance(data, Mapping):
            raise TypeError("state_dict must be a mapping")
        expected = {
            "agent",
            "epsilon",
            "hyperparameters",
            "q_table",
            "schema_version",
            *extra_keys,
        }
        _expect_keys(data, expected, "state_dict")
        version = _expect_int(data["schema_version"], "state_dict.schema_version")
        if version != TABULAR_AGENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported tabular-agent schema version: {version}")
        agent_name = _expect_string(data["agent"], "state_dict.agent")
        if agent_name != self.name:
            raise ValueError(f"state belongs to {agent_name!r}, not {self.name!r}")

        hyperparameters = _expect_mapping(
            data["hyperparameters"], "state_dict.hyperparameters"
        )
        expected_hyperparameters = self._hyperparameters()
        _expect_keys(
            hyperparameters,
            set(expected_hyperparameters),
            "state_dict.hyperparameters",
        )
        for key, expected_value in expected_hyperparameters.items():
            value = hyperparameters[key]
            if isinstance(expected_value, int):
                actual: JsonScalar = _expect_int(
                    value, f"state_dict.hyperparameters.{key}"
                )
            else:
                actual = _finite_float(value, f"state_dict.hyperparameters.{key}")
            if actual != expected_value:
                raise ValueError(f"state_dict hyperparameter mismatch: {key}")

        epsilon = _finite_float(data["epsilon"], "state_dict.epsilon")
        if not self.epsilon_min <= epsilon <= self.epsilon_initial:
            raise ValueError("state_dict epsilon is outside the configured schedule")
        raw_rows = data["q_table"]
        if not isinstance(raw_rows, list):
            raise TypeError("state_dict.q_table must be a JSON array")
        q_table: dict[TabularState, list[float]] = {}
        for index, raw_row in enumerate(raw_rows):
            path = f"state_dict.q_table[{index}]"
            row = _expect_mapping(raw_row, path)
            _expect_keys(row, {"state", "values"}, path)
            state = _json_state(row["state"], f"{path}.state")
            if state in q_table:
                raise ValueError("state_dict.q_table contains duplicate states")
            raw_values = row["values"]
            if not isinstance(raw_values, list) or len(raw_values) != len(ACTIONS):
                raise ValueError(f"{path}.values must contain one value per action")
            q_table[state] = [
                _finite_float(value, f"{path}.values[{value_index}]")
                for value_index, value in enumerate(raw_values)
            ]
        return q_table, epsilon


class DynaQAgent(QLearningAgent):
    """Q-learning plus seeded updates from a deterministic experience model."""

    name = "dyna-q"

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        guidance_probability: float = 0.0,
        progress_reward: float = 0.0,
        planning_steps: int = 10,
    ) -> None:
        super().__init__(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            guidance_probability=guidance_probability,
            progress_reward=progress_reward,
        )
        if isinstance(planning_steps, bool) or not isinstance(planning_steps, int):
            raise TypeError("planning_steps must be an integer")
        if planning_steps < 0:
            raise ValueError("planning_steps must be non-negative")
        self.planning_steps = planning_steps
        self._model: dict[tuple[TabularState, int], _ModelEntry] = {}
        self._model_keys: list[tuple[TabularState, int]] = []

    @property
    def model_size(self) -> int:
        return len(self._model)

    def clear(self) -> None:
        super().clear()
        self._model.clear()
        self._model_keys.clear()

    def _after_real_experience(self, experience: _Experience) -> None:
        key = (experience.state, experience.action_index)
        if key not in self._model:
            self._model_keys.append(key)
        self._model[key] = _ModelEntry(
            reward=experience.reward,
            next_state=experience.next_state,
            terminated=experience.terminated,
            next_actions=experience.next_actions,
        )
        for _ in range(self.planning_steps):
            sampled_key = self._model_keys[int(self._rng.integers(len(self._model_keys)))]
            state, action_index = sampled_key
            entry = self._model[sampled_key]
            self._q_update(
                _Experience(
                    state=state,
                    action_index=action_index,
                    reward=entry.reward,
                    next_state=entry.next_state,
                    terminated=entry.terminated,
                    next_actions=entry.next_actions,
                )
            )
        self._record_learning_update(self.planning_steps)

    def _hyperparameters(self) -> dict[str, JsonScalar]:
        hyperparameters = super()._hyperparameters()
        hyperparameters["planning_steps"] = self.planning_steps
        return hyperparameters

    def state_dict(self) -> dict[str, object]:
        data = super().state_dict()
        model_rows: list[dict[str, object]] = []
        for (state, action_index), entry in sorted(self._model.items()):
            model_rows.append(
                {
                    "action": ACTIONS[action_index].value,
                    "next_actions": [ACTIONS[index].value for index in entry.next_actions],
                    "next_state": list(entry.next_state),
                    "reward": entry.reward,
                    "state": list(state),
                    "terminated": entry.terminated,
                }
            )
        data["model"] = model_rows
        return data

    def load_state_dict(self, data: Mapping[str, object]) -> None:
        q_table, epsilon = self._decode_base_state(
            data, extra_keys=frozenset({"model"})
        )
        raw_model = data["model"]
        if not isinstance(raw_model, list):
            raise TypeError("state_dict.model must be a JSON array")
        model: dict[tuple[TabularState, int], _ModelEntry] = {}
        for index, raw_row in enumerate(raw_model):
            path = f"state_dict.model[{index}]"
            row = _expect_mapping(raw_row, path)
            _expect_keys(
                row,
                {
                    "action",
                    "next_actions",
                    "next_state",
                    "reward",
                    "state",
                    "terminated",
                },
                path,
            )
            action = _expect_action(row["action"], f"{path}.action")
            key = (_json_state(row["state"], f"{path}.state"), ACTION_INDEX[action])
            if key in model:
                raise ValueError("state_dict.model contains duplicate state-action pairs")
            raw_next_actions = row["next_actions"]
            if not isinstance(raw_next_actions, list):
                raise TypeError(f"{path}.next_actions must be a JSON array")
            next_actions = tuple(
                ACTION_INDEX[
                    _expect_action(value, f"{path}.next_actions[{action_index}]")
                ]
                for action_index, value in enumerate(raw_next_actions)
            )
            if len(set(next_actions)) != len(next_actions):
                raise ValueError(f"{path}.next_actions contains duplicates")
            terminated = row["terminated"]
            if not isinstance(terminated, bool):
                raise TypeError(f"{path}.terminated must be a boolean")
            model[key] = _ModelEntry(
                reward=_finite_float(row["reward"], f"{path}.reward"),
                next_state=_json_state(row["next_state"], f"{path}.next_state"),
                terminated=terminated,
                next_actions=tuple(sorted(next_actions)),
            )
        self._q_table = q_table
        self._epsilon = epsilon
        self._model = model
        self._model_keys = sorted(model)
        self._pending_snapshot = None
        self._pending_action = None


def _probability(name: str, value: float, *, positive: bool = False) -> float:
    numeric = _finite_float(value, name)
    lower_ok = numeric > 0.0 if positive else numeric >= 0.0
    if not lower_ok or numeric > 1.0:
        bracket = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must be in {bracket}")
    return numeric


def _finite_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{path} must be finite")
    return numeric


def _runtime_state(value: object, path: str) -> TabularState:
    if not isinstance(value, tuple):
        raise TypeError(f"{path} must be a tuple of integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{path} must contain only integers")
    return cast(TabularState, value)


def _json_state(value: object, path: str) -> TabularState:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{path} must contain only integers")
    return tuple(cast(list[int], value))


def _expect_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings")
    return cast(Mapping[str, object], value)


def _expect_keys(data: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"invalid {path} keys; missing={missing}, unknown={unknown}")


def _expect_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    return value


def _expect_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    if not value:
        raise ValueError(f"{path} must not be empty")
    return value


def _expect_action(value: object, path: str) -> Action:
    raw = _expect_string(value, path)
    try:
        return Action(raw)
    except ValueError as error:
        raise ValueError(f"{path} contains an unknown action: {raw!r}") from error


__all__ = [
    "TABULAR_AGENT_SCHEMA_VERSION",
    "DynaQAgent",
    "QLearningAgent",
    "QValues",
]

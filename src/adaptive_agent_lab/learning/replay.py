"""Deterministic, environment-independent experience replay."""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True, slots=True)
class Transition(Generic[StateT, ActionT]):
    """One immutable transition record stored by :class:`ReplayBuffer`."""

    state: StateT
    action: ActionT
    reward: float
    next_state: StateT
    done: bool

    @property
    def terminated(self) -> bool:
        """Expose Gym-style terminology without coupling to Gym itself."""

        return self.done


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """A numeric batch ready for a value-learning update."""

    states: npt.NDArray[np.float64]
    actions: npt.NDArray[np.int64]
    rewards: npt.NDArray[np.float64]
    next_states: npt.NDArray[np.float64]
    dones: npt.NDArray[np.bool_]


class ReplayBuffer(Generic[StateT, ActionT]):
    """A fixed-capacity FIFO replay buffer with seeded sampling.

    Samples are drawn without replacement.  Stored values are defensively
    copied on insertion and sampling so caller mutation cannot change history.
    """

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self._capacity = capacity
        self._seed = seed
        self._rng = random.Random(seed)
        self._storage: list[Transition[StateT, ActionT] | None] = [None] * capacity
        self._size = 0
        self._next_index = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return self._size == self._capacity

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: StateT,
        action: ActionT,
        reward: float,
        next_state: StateT,
        done: bool,
    ) -> None:
        """Store a transition, replacing the oldest item when full."""

        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise TypeError("reward must be a real number")
        numeric_reward = float(reward)
        if not np.isfinite(numeric_reward):
            raise ValueError("reward must be finite")
        if not isinstance(done, (bool, np.bool_)):
            raise TypeError("done must be a boolean")
        transition = Transition(
            copy.deepcopy(state),
            copy.deepcopy(action),
            numeric_reward,
            copy.deepcopy(next_state),
            bool(done),
        )
        self.append(transition)

    def push(
        self,
        state: StateT,
        action: ActionT,
        reward: float,
        next_state: StateT,
        done: bool,
    ) -> None:
        """Alias for :meth:`add`, matching common replay-buffer terminology."""

        self.add(state, action, reward, next_state, done)

    def append(self, transition: Transition[StateT, ActionT]) -> None:
        """Append an already constructed transition."""

        if not isinstance(transition, Transition):
            raise TypeError("transition must be a Transition")
        if isinstance(transition.reward, bool) or not isinstance(transition.reward, Real):
            raise TypeError("transition reward must be a real number")
        if not np.isfinite(float(transition.reward)):
            raise ValueError("transition reward must be finite")
        if not isinstance(transition.done, (bool, np.bool_)):
            raise TypeError("transition done flag must be a boolean")
        stored = Transition(
            copy.deepcopy(transition.state),
            copy.deepcopy(transition.action),
            float(transition.reward),
            copy.deepcopy(transition.next_state),
            bool(transition.done),
        )
        self._storage[self._next_index] = stored
        self._next_index = (self._next_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> tuple[Transition[StateT, ActionT], ...]:
        """Return a deterministic seeded sample without replacement."""

        return tuple(copy.deepcopy(item) for item in self._sample_refs(batch_size))

    def sample_batch(self, batch_size: int) -> ReplayBatch:
        """Sample and stack numeric transitions into homogeneous NumPy arrays."""

        # Stacking creates independent arrays, so this numeric fast path can
        # avoid deep-copying every selected transition first.
        transitions = self._sample_refs(batch_size)
        try:
            states = np.stack(
                [np.asarray(item.state, dtype=np.float64) for item in transitions]
            )
            next_states = np.stack(
                [np.asarray(item.next_state, dtype=np.float64) for item in transitions]
            )
            raw_actions = [item.action for item in transitions]
            invalid_action = any(
                isinstance(action, bool) or not isinstance(action, Integral)
                for action in raw_actions
            )
            if invalid_action:
                raise ValueError("sampled actions must be scalar integer indices")
            actions = np.asarray(raw_actions, dtype=np.int64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "sampled states and actions must form homogeneous numeric arrays"
            ) from error
        if actions.ndim != 1:
            raise ValueError("sampled actions must be scalar integer indices")
        return ReplayBatch(
            states=states,
            actions=actions,
            rewards=np.asarray([item.reward for item in transitions], dtype=np.float64),
            next_states=next_states,
            dones=np.asarray([item.done for item in transitions], dtype=np.bool_),
        )

    def transitions(self) -> tuple[Transition[StateT, ActionT], ...]:
        """Return all stored transitions from oldest to newest."""

        indices: Iterable[int]
        if self._size < self._capacity:
            indices = range(self._size)
        else:
            indices = (
                *range(self._next_index, self._capacity),
                *range(0, self._next_index),
            )
        result: list[Transition[StateT, ActionT]] = []
        for index in indices:
            transition = self._storage[index]
            if transition is None:  # Defensive guard for internal invariants.
                raise RuntimeError("replay buffer storage is inconsistent")
            result.append(copy.deepcopy(transition))
        return tuple(result)

    def clear(self, *, reset_rng: bool = False) -> None:
        """Remove transitions, optionally restoring the initial sampling stream."""

        self._storage = [None] * self._capacity
        self._size = 0
        self._next_index = 0
        if reset_rng:
            self._rng.seed(self._seed)

    def _validate_batch_size(self, batch_size: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > self._size:
            raise ValueError(
                f"cannot sample {batch_size} transitions from a buffer containing {self._size}"
            )

    def _sample_refs(
        self, batch_size: int
    ) -> tuple[Transition[StateT, ActionT], ...]:
        self._validate_batch_size(batch_size)
        logical_indices = self._rng.sample(range(self._size), batch_size)
        sampled: list[Transition[StateT, ActionT]] = []
        for logical_index in logical_indices:
            storage_index = (
                logical_index
                if self._size < self._capacity
                else (self._next_index + logical_index) % self._capacity
            )
            transition = self._storage[storage_index]
            if transition is None:
                raise RuntimeError("replay buffer storage is inconsistent")
            sampled.append(transition)
        return tuple(sampled)


__all__ = ["ReplayBatch", "ReplayBuffer", "Transition"]

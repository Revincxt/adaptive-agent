"""Learning-augmented planning with semi-MDP option learning."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from adaptive_agent_lab.agents.base import Agent
from adaptive_agent_lab.agents.planning import SearchResult, astar_path
from adaptive_agent_lab.environment.contracts import (
    Action,
    Order,
    OrderStatus,
    Position,
    Transition,
    WarehouseSnapshot,
    WarehouseState,
)
from adaptive_agent_lab.environment.observation import (
    ACTION_INDEX,
    ObservationEncoder,
    ObservationSpec,
)
from adaptive_agent_lab.learning.network import MLPQNetwork

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """High-level option learner and low-level safety-planner settings."""

    hidden: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon: float = 0.10
    battery_reserve: int = 2
    max_grad_norm: float = 10.0

    def __post_init__(self) -> None:
        if len(self.hidden) not in (1, 2) or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in self.hidden
        ):
            raise ValueError("hidden must contain one or two positive integer widths")
        for name in ("gamma", "lr", "epsilon", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.lr <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("lr and max_grad_norm must be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if (
            isinstance(self.battery_reserve, bool)
            or not isinstance(self.battery_reserve, int)
            or self.battery_reserve < 0
        ):
            raise ValueError("battery_reserve must be a non-negative integer")

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        """Compatibility alias matching the lower-level network terminology."""

        return self.hidden

    @property
    def learning_rate(self) -> float:
        return float(self.lr)


class HybridAgent(Agent):
    """Choose task options with a Q network and execute them through A*.

    An order option spans both pickup and drop-off.  Its option identity remains
    fixed while a dynamic closure causes the low-level path to be repaired.
    Only primitive :class:`Action` values are returned; this class never mutates
    environment state or bypasses the shared transition validator.
    """

    name = "hybrid"
    trainable = True

    def __init__(self, config: HybridConfig | None = None) -> None:
        super().__init__()
        config = HybridConfig() if config is None else config
        if not isinstance(config, HybridConfig):
            raise TypeError("config must be a HybridConfig")
        self.config = config
        self._encoder: ObservationEncoder | None = None
        self._template: WarehouseSnapshot | None = None
        self._network: MLPQNetwork | None = None
        self._active_option: int | None = None
        self._option_start: FloatArray | None = None
        self._option_return = 0.0
        self._option_duration = 0
        self._route: list[Action] = []
        self._route_signature: tuple[Position, Action | None, frozenset[Position]] | None = None
        self._awaiting_observation = False
        self._last_action: Action | None = None
        self._environment_steps = 0
        self._updates = 0
        self._last_loss: float | None = None
        self._last_td_target: float | None = None
        self._last_option_duration: int | None = None
        self._last_discount: float | None = None
        self._last_outcome: str | None = None

    @property
    def option_count(self) -> int:
        if self._template is not None:
            return len(self._template.orders) + 2
        if self._network is not None:
            return self._network.output_dim
        raise RuntimeError("agent must be reset or loaded before option_count is available")

    @property
    def update_count(self) -> int:
        return self._updates

    @property
    def environment_steps(self) -> int:
        return self._environment_steps

    @property
    def current_option_index(self) -> int | None:
        return self._active_option

    @property
    def current_option(self) -> str | None:
        if self._active_option is None or self._template is None:
            return None
        return self.option_labels(self._template)[self._active_option]

    @property
    def last_loss(self) -> float | None:
        return self._last_loss

    @property
    def last_td_target(self) -> float | None:
        return self._last_td_target

    @property
    def last_option_duration(self) -> int | None:
        return self._last_option_duration

    @property
    def last_discount(self) -> float | None:
        return self._last_discount

    @property
    def last_outcome(self) -> str | None:
        return self._last_outcome

    def _reset(self, snapshot: WarehouseSnapshot) -> None:
        spec = ObservationSpec.from_snapshot(snapshot)
        option_count = len(snapshot.orders) + 2
        if self._network is None:
            network_seed = int(self._rng.integers(0, 2**32 - 1))
            self._network = MLPQNetwork(
                spec.vector_size,
                option_count,
                self.config.hidden,
                seed=network_seed,
            )
        elif (
            self._network.input_dim != spec.vector_size
            or self._network.output_dim != option_count
        ):
            raise ValueError(
                "hybrid agent cannot retain a network across incompatible observation specs"
            )
        self._encoder = ObservationEncoder(spec)
        self._template = snapshot
        self._clear_active_option()
        self._awaiting_observation = False
        self._last_action = None

    def option_labels(self, snapshot: WarehouseSnapshot) -> tuple[str, ...]:
        return (
            *(f"order:{order.order_id}" for order in snapshot.orders),
            "charge",
            "wait",
        )

    def option_mask(self, snapshot: WarehouseSnapshot) -> BoolArray:
        encoder, _ = self._ready()
        encoder.vector(snapshot)
        return self._option_mask_unchecked(snapshot)

    def q_values(self, snapshot: WarehouseSnapshot) -> FloatArray:
        encoder, network = self._ready()
        return network.predict(encoder.vector(snapshot).astype(np.float64))

    def _option_mask_unchecked(self, snapshot: WarehouseSnapshot) -> BoolArray:
        count = len(snapshot.orders) + 2
        mask = np.zeros(count, dtype=np.bool_)
        if snapshot.state.terminated:
            return mask
        robot = snapshot.state.robot
        if robot.carried_order_id is None:
            for index, order in enumerate(snapshot.orders):
                mask[index] = (
                    snapshot.state.order_status[order.order_id] is OrderStatus.AVAILABLE
                )
        charge_index = len(snapshot.orders)
        if robot.battery < snapshot.battery_capacity:
            mask[charge_index] = any(
                route.reached and route.cost <= robot.battery
                for route in (
                    astar_path(
                        snapshot.map,
                        robot.position,
                        charger,
                        blocked_cells=snapshot.state.blocked_cells,
                    )
                    for charger in sorted(snapshot.map.charging_stations)
                )
            )
        mask[charge_index + 1] = True
        return mask

    def _act(self, snapshot: WarehouseSnapshot, *, explore: bool) -> Action:
        if self._awaiting_observation:
            assert self._last_action is not None
            return self._last_action
        if self._active_option is None:
            self._start_option(snapshot, self._select_option(snapshot, explore=explore))

        action = self._primitive_action(snapshot)
        encoder, _ = self._ready()
        if not encoder.action_mask(snapshot)[ACTION_INDEX[action]]:
            self._clear_route()
            action = Action.WAIT
        self._last_action = action
        self._awaiting_observation = True
        return action

    def _select_option(self, snapshot: WarehouseSnapshot, *, explore: bool) -> int:
        encoder, network = self._ready()
        vector = encoder.vector(snapshot).astype(np.float64)
        mask = self._option_mask_unchecked(snapshot)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            raise RuntimeError("non-terminal snapshots must have at least the wait option")
        if explore and self._rng.random() < self.config.epsilon:
            return int(self._rng.choice(valid))
        values = network.predict(vector)
        return int(np.argmax(np.where(mask, values, -np.inf)))

    def _start_option(self, snapshot: WarehouseSnapshot, option_index: int) -> None:
        if not 0 <= option_index < len(snapshot.orders) + 2:
            raise ValueError("option index is outside the current option space")
        encoder, _ = self._ready()
        self._active_option = option_index
        self._option_start = encoder.vector(snapshot).astype(np.float64)
        self._option_return = 0.0
        self._option_duration = 0
        self._clear_route()

    def _primitive_action(self, snapshot: WarehouseSnapshot) -> Action:
        if self._active_option is None:
            raise RuntimeError("no active option")
        order_count = len(snapshot.orders)
        if self._active_option < order_count:
            return self._order_action(snapshot, snapshot.orders[self._active_option])
        if self._active_option == order_count:
            return self._charge_action(snapshot)
        return Action.WAIT

    def _order_action(self, snapshot: WarehouseSnapshot, order: Order) -> Action:
        state = snapshot.state
        status = state.order_status[order.order_id]
        if state.robot.carried_order_id == order.order_id or status is OrderStatus.PICKED_UP:
            goal = order.dropoff
            service_action = Action.DROPOFF
        elif status is OrderStatus.AVAILABLE and state.robot.carried_order_id is None:
            goal = order.pickup
            service_action = Action.PICKUP
        else:
            return Action.WAIT

        route = astar_path(
            snapshot.map,
            state.robot.position,
            goal,
            blocked_cells=state.blocked_cells,
        )
        if not route.reached:
            self._clear_route()
            return Action.WAIT
        required_battery = (
            route.cost
            + self._post_service_energy(snapshot, order, service_action)
            + self.config.battery_reserve
        )
        if state.robot.battery < required_battery:
            if (
                state.robot.position in snapshot.map.charging_stations
                and state.robot.battery < snapshot.battery_capacity
            ):
                self._clear_route()
                return Action.CHARGE
            charger = self._nearest_reachable_charger(snapshot)
            if charger is None:
                self._clear_route()
                return Action.WAIT
            return self._planned_action(snapshot, charger[0], None)
        return self._planned_action(snapshot, goal, service_action)

    def _charge_action(self, snapshot: WarehouseSnapshot) -> Action:
        robot = snapshot.state.robot
        if robot.battery >= snapshot.battery_capacity:
            return Action.WAIT
        if robot.position in snapshot.map.charging_stations:
            self._clear_route()
            return Action.CHARGE
        charger = self._nearest_reachable_charger(snapshot)
        if charger is None:
            self._clear_route()
            return Action.WAIT
        return self._planned_action(snapshot, charger[0], None)

    def _nearest_reachable_charger(
        self, snapshot: WarehouseSnapshot
    ) -> tuple[Position, SearchResult] | None:
        candidates: list[tuple[int, Position, SearchResult]] = []
        for charger in sorted(snapshot.map.charging_stations):
            route = astar_path(
                snapshot.map,
                snapshot.state.robot.position,
                charger,
                blocked_cells=snapshot.state.blocked_cells,
            )
            if route.reached and route.cost <= snapshot.state.robot.battery:
                candidates.append((route.cost, charger, route))
        if not candidates:
            return None
        _, position, route = min(candidates, key=lambda item: (item[0], item[1]))
        return position, route

    def _post_service_energy(
        self,
        snapshot: WarehouseSnapshot,
        order: Order,
        service_action: Action,
    ) -> int:
        position = order.dropoff if service_action is Action.DROPOFF else order.pickup
        cost = 0
        if service_action is Action.PICKUP:
            delivery = astar_path(
                snapshot.map,
                order.pickup,
                order.dropoff,
                blocked_cells=snapshot.state.blocked_cells,
            )
            if not delivery.reached:
                return snapshot.battery_capacity + 1
            cost = delivery.cost
            position = order.dropoff
        charger_distances = [
            route.cost
            for charger in sorted(snapshot.map.charging_stations)
            if (
                route := astar_path(
                    snapshot.map,
                    position,
                    charger,
                    blocked_cells=snapshot.state.blocked_cells,
                )
            ).reached
        ]
        return cost + (min(charger_distances) if charger_distances else 0)

    def _planned_action(
        self,
        snapshot: WarehouseSnapshot,
        goal: Position,
        service_action: Action | None,
    ) -> Action:
        signature = (goal, service_action, snapshot.state.blocked_cells)
        if not self._route or self._route_signature != signature:
            result = astar_path(
                snapshot.map,
                snapshot.state.robot.position,
                goal,
                blocked_cells=snapshot.state.blocked_cells,
            )
            self._record_plan(result.expanded_nodes)
            self._route_signature = signature
            if not result.reached:
                self._route = []
                return Action.WAIT
            self._route = list(result.actions)
            if service_action is not None:
                self._route.append(service_action)
        if not self._route:
            return Action.WAIT
        return self._route.pop(0)

    def observe(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise TypeError("transition must be a Transition")
        if not self._awaiting_observation or self._last_action is None:
            raise RuntimeError("act must be called before observe")
        if transition.action is not self._last_action:
            raise ValueError("observed transition action does not match the selected action")
        if self._active_option is None or self._option_start is None:
            raise RuntimeError("an observed action must belong to an active option")

        step_duration = transition.next_state.time - transition.state.time
        if step_duration < 1:
            raise ValueError("option transitions must advance time")
        self._option_return += (
            self.config.gamma**self._option_duration * transition.reward
        )
        self._option_duration += step_duration
        if self.learning_enabled:
            self._environment_steps += step_duration
        self._awaiting_observation = False
        self._last_action = None

        next_snapshot = self._snapshot_with_state(transition.next_state)
        outcome = self._option_outcome(next_snapshot, transition)
        if outcome is not None:
            self._finish_option(
                next_snapshot,
                terminal=transition.terminated,
                outcome=outcome,
            )

    def end_episode(self, snapshot: WarehouseSnapshot) -> None:
        if self._active_option is not None and self._option_duration > 0:
            self._finish_option(snapshot, terminal=True, outcome="terminal")
        else:
            self._clear_active_option()
        self._awaiting_observation = False
        self._last_action = None

    def _option_outcome(
        self,
        snapshot: WarehouseSnapshot,
        transition: Transition,
    ) -> str | None:
        if transition.terminated:
            return "terminal"
        if transition.violations:
            return "failure"
        assert self._active_option is not None
        order_count = len(snapshot.orders)
        if self._active_option < order_count:
            status = snapshot.state.order_status[
                snapshot.orders[self._active_option].order_id
            ]
            if status is OrderStatus.DELIVERED:
                return "complete"
            if status in {OrderStatus.EXPIRED, OrderStatus.PENDING}:
                return "failure"
            return None
        if self._active_option == order_count:
            if snapshot.state.robot.battery >= snapshot.battery_capacity:
                return "complete"
            return None
        return "complete"

    def _finish_option(
        self,
        next_snapshot: WarehouseSnapshot,
        *,
        terminal: bool,
        outcome: str,
    ) -> None:
        if (
            self._active_option is None
            or self._option_start is None
            or self._option_duration < 1
        ):
            raise RuntimeError("cannot finish an option without an observed duration")
        network = self._require_network()
        bootstrap = 0.0
        if not terminal:
            mask = self._option_mask_unchecked(next_snapshot)
            if np.any(mask):
                encoder, _ = self._ready()
                next_vector = encoder.vector(next_snapshot).astype(np.float64)
                bootstrap = float(np.max(np.where(mask, network.predict(next_vector), -np.inf)))
        discount = self.config.gamma**self._option_duration
        target = self._option_return + discount * bootstrap
        if self.learning_enabled:
            self._last_loss = network.train_td_batch(
                self._option_start.reshape(1, -1),
                np.asarray([self._active_option], dtype=np.int64),
                np.asarray([target], dtype=np.float64),
                learning_rate=self.config.lr,
                max_grad_norm=self.config.max_grad_norm,
            )
            self._updates += 1
            self._record_learning_update()
        self._last_td_target = target
        self._last_option_duration = self._option_duration
        self._last_discount = discount
        self._last_outcome = outcome
        self._clear_active_option()

    def state_dict(self) -> dict[str, object]:
        network = self._require_network()
        return {
            "format_version": 1,
            "agent": self.name,
            "config": {**asdict(self.config), "hidden": list(self.config.hidden)},
            "environment_steps": self._environment_steps,
            "updates": self._updates,
            "network": network.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("format_version") != 1 or state.get("agent") != self.name:
            raise ValueError("unsupported hybrid agent state")
        raw_network = state.get("network")
        if not isinstance(raw_network, Mapping):
            raise ValueError("hybrid state must contain a network mapping")
        network = MLPQNetwork.from_state_dict(raw_network)
        if network.hidden_sizes != self.config.hidden:
            raise ValueError("network hidden layers do not match the hybrid config")
        self._network = network
        self._environment_steps = _state_nonnegative_int(state, "environment_steps")
        self._updates = _state_nonnegative_int(state, "updates")
        self._clear_active_option()
        self._awaiting_observation = False
        self._last_action = None

    def clear_learning(self) -> None:
        self._network = None
        self._environment_steps = 0
        self._updates = 0
        self._last_loss = None
        self._last_td_target = None
        self._last_option_duration = None
        self._last_discount = None
        self._last_outcome = None
        self._clear_active_option()

    def _snapshot_with_state(self, state: WarehouseState) -> WarehouseSnapshot:
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
            raise RuntimeError("hybrid network is not initialized")
        return self._network

    def _clear_route(self) -> None:
        self._route = []
        self._route_signature = None

    def _clear_active_option(self) -> None:
        self._active_option = None
        self._option_start = None
        self._option_return = 0.0
        self._option_duration = 0
        self._clear_route()


def _state_nonnegative_int(state: Mapping[str, object], key: str) -> int:
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def config_from_mapping(data: Mapping[str, object]) -> HybridConfig:
    """Construct a validated config from JSON-compatible data."""

    raw_hidden = data.get("hidden", data.get("hidden_sizes", (64, 64)))
    if not isinstance(raw_hidden, Sequence) or isinstance(raw_hidden, (str, bytes)):
        raise TypeError("hidden must be an array")
    return HybridConfig(
        hidden=tuple(int(value) for value in raw_hidden),
        gamma=float(cast(str | int | float, data.get("gamma", 0.99))),
        lr=float(
            cast(str | int | float, data.get("lr", data.get("learning_rate", 1e-3)))
        ),
        epsilon=float(cast(str | int | float, data.get("epsilon", 0.10))),
        battery_reserve=int(cast(str | int | float, data.get("battery_reserve", 2))),
        max_grad_norm=float(cast(str | int | float, data.get("max_grad_norm", 10.0))),
    )


__all__ = ["HybridAgent", "HybridConfig", "config_from_mapping"]

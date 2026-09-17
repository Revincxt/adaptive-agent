"""Shared tabular, neural, and action-mask views of warehouse snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from adaptive_agent_lab.environment.contracts import (
    Action,
    OrderStatus,
    Position,
    WarehouseSnapshot,
)

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
TabularState = tuple[int, ...]

ACTIONS: tuple[Action, ...] = tuple(Action)
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}
STATUS_INDEX = {status: index for index, status in enumerate(OrderStatus)}


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Fixed dimensions and normalization constants for one training suite."""

    width: int
    height: int
    max_orders: int
    battery_capacity: int
    horizon: int
    max_priority: float

    @classmethod
    def from_snapshot(
        cls,
        snapshot: WarehouseSnapshot,
        *,
        max_orders: int | None = None,
    ) -> ObservationSpec:
        order_capacity = len(snapshot.orders) if max_orders is None else max_orders
        if order_capacity < len(snapshot.orders):
            raise ValueError("max_orders cannot be smaller than the scenario order count")
        return cls(
            width=snapshot.map.width,
            height=snapshot.map.height,
            max_orders=order_capacity,
            battery_capacity=snapshot.battery_capacity,
            horizon=snapshot.horizon,
            max_priority=max((order.priority for order in snapshot.orders), default=1.0),
        )

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("map dimensions must be positive")
        if self.max_orders < 0:
            raise ValueError("max_orders must be non-negative")
        if self.battery_capacity < 1 or self.horizon < 1 or self.max_priority <= 0:
            raise ValueError("normalization constants must be positive")

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    @property
    def vector_size(self) -> int:
        # Four grid channels; six global scalars; then seven order values and
        # five status bits per padded order slot.
        return 4 * self.cell_count + 6 + 12 * self.max_orders


class ObservationEncoder:
    """Encode snapshots without giving any agent privileged environment state."""

    def __init__(self, spec: ObservationSpec) -> None:
        if not isinstance(spec, ObservationSpec):
            raise TypeError("spec must be an ObservationSpec")
        self.spec = spec

    def _check(self, snapshot: WarehouseSnapshot) -> None:
        if snapshot.map.width != self.spec.width or snapshot.map.height != self.spec.height:
            raise ValueError("snapshot map dimensions do not match the observation spec")
        if len(snapshot.orders) > self.spec.max_orders:
            raise ValueError("snapshot contains more orders than the observation spec")
        if snapshot.battery_capacity != self.spec.battery_capacity:
            raise ValueError("snapshot battery capacity does not match the observation spec")
        if snapshot.horizon != self.spec.horizon:
            raise ValueError("snapshot horizon does not match the observation spec")

    def vector(self, snapshot: WarehouseSnapshot) -> FloatArray:
        """Return a padded, normalized vector suitable for the NumPy DQN."""

        self._check(snapshot)
        cell_count = self.spec.cell_count
        vector = np.zeros(self.spec.vector_size, dtype=np.float32)
        obstacle_start = 0
        charger_start = cell_count
        blocked_start = 2 * cell_count
        robot_start = 3 * cell_count
        for position in snapshot.map.obstacles:
            vector[obstacle_start + self._cell_index(position)] = 1.0
        for position in snapshot.map.charging_stations:
            vector[charger_start + self._cell_index(position)] = 1.0
        for position in snapshot.state.blocked_cells:
            vector[blocked_start + self._cell_index(position)] = 1.0
        vector[robot_start + self._cell_index(snapshot.state.robot.position)] = 1.0

        offset = 4 * cell_count
        vector[offset] = snapshot.state.time / self.spec.horizon
        vector[offset + 1] = snapshot.state.robot.battery / self.spec.battery_capacity
        vector[offset + 2] = float(snapshot.state.robot.carried_order_id is not None)

        width_scale = max(self.spec.width - 1, 1)
        height_scale = max(self.spec.height - 1, 1)
        target = self._active_target(snapshot)
        if target is not None:
            robot = snapshot.state.robot.position
            vector[offset + 3] = (target.x - robot.x) / (2 * width_scale) + 0.5
            vector[offset + 4] = (target.y - robot.y) / (2 * height_scale) + 0.5
            vector[offset + 5] = 1.0
        offset += 6

        for order_index, order in enumerate(snapshot.orders):
            base = offset + 12 * order_index
            vector[base : base + 7] = (
                order.pickup.x / width_scale,
                order.pickup.y / height_scale,
                order.dropoff.x / width_scale,
                order.dropoff.y / height_scale,
                order.release_time / self.spec.horizon,
                order.deadline / self.spec.horizon,
                order.priority / self.spec.max_priority,
            )
            status_index = STATUS_INDEX[snapshot.state.order_status[order.order_id]]
            vector[base + 7 + status_index] = 1.0
        return vector

    @staticmethod
    def _active_target(snapshot: WarehouseSnapshot) -> Position | None:
        carried_id = snapshot.state.robot.carried_order_id
        if carried_id is not None:
            return snapshot.order_by_id(carried_id).dropoff
        available = [
            order
            for order in snapshot.orders
            if snapshot.state.order_status[order.order_id] is OrderStatus.AVAILABLE
        ]
        if not available:
            return None
        robot = snapshot.state.robot.position
        order = min(
            available,
            key=lambda candidate: (
                candidate.deadline,
                -candidate.priority,
                robot.manhattan_distance(candidate.pickup),
                candidate.order_id,
            ),
        )
        return order.pickup

    def tabular(self, snapshot: WarehouseSnapshot) -> TabularState:
        """Return an exact hashable state for controlled small-map experiments."""

        self._check(snapshot)
        carried_id = snapshot.state.robot.carried_order_id
        carried_index = 0
        if carried_id is not None:
            carried_index = next(
                index + 1
                for index, order in enumerate(snapshot.orders)
                if order.order_id == carried_id
            )
        blocked_mask = 0
        for position in snapshot.state.blocked_cells:
            blocked_mask |= 1 << self._cell_index(position)
        return (
            snapshot.state.time,
            self._cell_index(snapshot.state.robot.position),
            snapshot.state.robot.battery,
            carried_index,
            *(
                STATUS_INDEX[snapshot.state.order_status[order.order_id]]
                for order in snapshot.orders
            ),
            blocked_mask,
        )

    def action_mask(self, snapshot: WarehouseSnapshot) -> BoolArray:
        """Return actions that satisfy immediately checkable hard constraints."""

        self._check(snapshot)
        mask = np.zeros(len(ACTIONS), dtype=np.bool_)
        state = snapshot.state
        robot = state.robot
        if not state.terminated:
            mask[ACTION_INDEX[Action.WAIT]] = True
            for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
                destination = robot.position.moved(action)
                mask[ACTION_INDEX[action]] = (
                    robot.battery > 0
                    and snapshot.map.is_traversable(destination, state.blocked_cells)
                )
            mask[ACTION_INDEX[Action.PICKUP]] = (
                robot.carried_order_id is None
                and any(
                    order.pickup == robot.position
                    and state.order_status[order.order_id] is OrderStatus.AVAILABLE
                    for order in snapshot.orders
                )
            )
            if robot.carried_order_id is not None:
                order = snapshot.order_by_id(robot.carried_order_id)
                mask[ACTION_INDEX[Action.DROPOFF]] = robot.position == order.dropoff
            mask[ACTION_INDEX[Action.CHARGE]] = (
                robot.position in snapshot.map.charging_stations
                and robot.battery < snapshot.battery_capacity
            )
        return mask

    def _cell_index(self, position: Position) -> int:
        if not 0 <= position.x < self.spec.width or not 0 <= position.y < self.spec.height:
            raise ValueError(f"position outside observation grid: {position!r}")
        return position.y * self.spec.width + position.x


__all__ = [
    "ACTIONS",
    "ACTION_INDEX",
    "BoolArray",
    "FloatArray",
    "ObservationEncoder",
    "ObservationSpec",
    "TabularState",
]

from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

from adaptive_agent_lab.agents.planning import ReplanningAgent
from adaptive_agent_lab.benchmarking.runner import run_episode
from adaptive_agent_lab.environment.contracts import Position
from adaptive_agent_lab.environment.events import EventKind
from adaptive_agent_lab.environment.scenario import Scenario

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "scenarios" / "medium"
REFERENCE_PATH = SCENARIO_DIR / "maze-warehouse.json"
SCENARIO_PATHS = tuple(
    SCENARIO_DIR / filename
    for filename in ("parallel-aisles.json", "cross-dock.json", "serpentine.json")
)

EXPECTED_OBSTACLE_COUNTS = {
    "parallel-aisles": 42,
    "cross-dock": 48,
    "serpentine": 48,
}

# Each witness crosses its closure in two steps while the cell is open. Closing
# that cell must preserve connectivity but force the route around an aisle end.
DETOUR_WITNESSES = {
    "parallel-aisles": (
        (Position(5, 5), Position(5, 4), Position(5, 6)),
        (Position(10, 9), Position(9, 9), Position(11, 9)),
    ),
    "cross-dock": (
        (Position(7, 5), Position(6, 5), Position(8, 5)),
        (Position(10, 10), Position(9, 10), Position(11, 10)),
    ),
    "serpentine": (
        (Position(5, 9), Position(5, 8), Position(5, 10)),
        (Position(10, 2), Position(10, 1), Position(10, 3)),
    ),
}


def _load(path: Path) -> tuple[str, Scenario]:
    text = path.read_text(encoding="utf-8")
    return text, Scenario.from_json(text)


def _reachable(
    scenario: Scenario,
    *,
    start: Position | None = None,
    blocked: frozenset[Position] = frozenset(),
) -> set[Position]:
    reached = {start or scenario.initial_robot.position}
    queue = deque(reached)
    while queue:
        current = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = current.translated(dx, dy)
            if scenario.map.is_traversable(neighbor, blocked) and neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def _shortest_path_length(
    scenario: Scenario,
    start: Position,
    goal: Position,
    *,
    blocked: frozenset[Position] = frozenset(),
) -> int | None:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            return distances[current]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = current.translated(dx, dy)
            if scenario.map.is_traversable(neighbor, blocked) and neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return None


@pytest.mark.parametrize("scenario_path", SCENARIO_PATHS, ids=lambda path: path.stem)
def test_gallery_fixture_is_canonical_and_matches_reference_workload(
    scenario_path: Path,
) -> None:
    text, scenario = _load(scenario_path)
    _, reference = _load(REFERENCE_PATH)

    assert text == scenario.to_json(indent=2) + "\n"
    assert scenario.scenario_id == f"{scenario_path.stem}-demo"
    assert (scenario.map.width, scenario.map.height) == (16, 12)
    assert scenario.horizon == reference.horizon == 240
    assert scenario.battery_capacity == reference.battery_capacity == 96
    assert scenario.initial_robot == reference.initial_robot
    assert scenario.map.charging_stations == reference.map.charging_stations
    assert len(scenario.orders) == len(reference.orders) == 6
    assert len(scenario.map.obstacles) == EXPECTED_OBSTACLE_COUNTS[scenario_path.stem]

    workload = [
        (
            order.order_id,
            order.dropoff,
            order.release_time,
            order.deadline,
            order.priority,
        )
        for order in scenario.orders
    ]
    reference_workload = [
        (
            order.order_id,
            order.dropoff,
            order.release_time,
            order.deadline,
            order.priority,
        )
        for order in reference.orders
    ]
    assert workload == reference_workload
    assert [
        (event.time, event.kind, event.order_id) for event in scenario.events
    ] == [
        (event.time, event.kind, event.order_id) for event in reference.events
    ]

    for order in scenario.orders:
        rack_neighbors = {
            order.pickup.translated(dx, dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        }
        assert rack_neighbors & scenario.map.obstacles
        assert order.dropoff.x >= 13
        assert order.dropoff.y >= 10


@pytest.mark.parametrize("scenario_path", SCENARIO_PATHS, ids=lambda path: path.stem)
def test_gallery_closures_are_paired_and_leave_real_detours(
    scenario_path: Path,
) -> None:
    _, scenario = _load(scenario_path)
    witnesses = DETOUR_WITNESSES[scenario_path.stem]
    expected_closures = [witness[0] for witness in witnesses]
    blocked_events = [event for event in scenario.events if event.kind is EventKind.CELL_BLOCKED]
    unblocked_events = [
        event for event in scenario.events if event.kind is EventKind.CELL_UNBLOCKED
    ]

    assert [(event.time, event.position) for event in blocked_events] == [
        (6, expected_closures[0]),
        (55, expected_closures[1]),
    ]
    assert [(event.time, event.position) for event in unblocked_events] == [
        (20, expected_closures[0]),
        (70, expected_closures[1]),
    ]

    required = set(scenario.map.charging_stations)
    required.update(order.pickup for order in scenario.orders)
    required.update(order.dropoff for order in scenario.orders)
    assert required <= _reachable(scenario)

    for closure, start, goal in witnesses:
        blocked = frozenset({closure})
        assert required <= _reachable(scenario, blocked=blocked)
        direct_distance = _shortest_path_length(scenario, start, goal)
        detour_distance = _shortest_path_length(
            scenario,
            start,
            goal,
            blocked=blocked,
        )
        assert direct_distance == 2
        assert detour_distance is not None
        assert detour_distance > direct_distance


@pytest.mark.parametrize("scenario_path", SCENARIO_PATHS, ids=lambda path: path.stem)
def test_replanning_completes_gallery_scenario_without_violations(
    scenario_path: Path,
) -> None:
    _, scenario = _load(scenario_path)
    result = run_episode(
        ReplanningAgent(),
        scenario,
        seed=42,
        explore=False,
        learn=False,
        measure_timing=False,
    )

    assert result.metrics.completed_orders == len(scenario.orders)
    assert result.metrics.weighted_on_time_completion_rate >= 2 / 3
    assert result.metrics.constraint_violations == 0

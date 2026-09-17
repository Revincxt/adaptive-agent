"""Build the browser replay artifact from real environment trajectories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from adaptive_agent_lab.agents.base import Agent
from adaptive_agent_lab.agents.dqn import DQNAgent, DQNConfig
from adaptive_agent_lab.agents.hybrid import HybridAgent, HybridConfig
from adaptive_agent_lab.agents.planning import OpenLoopPlanningAgent, ReplanningAgent
from adaptive_agent_lab.agents.tabular import DynaQAgent, QLearningAgent
from adaptive_agent_lab.benchmarking.runner import EpisodeResult, run_episode, train_episodes
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.reporting.artifacts import fingerprint, write_json_atomic

DEFAULT_TRAINING_EPISODES: Mapping[str, int] = {
    "q-learning": 120,
    "dyna-q": 80,
    "dqn": 80,
    "hybrid": 40,
}

DEMO_GALLERY_SCHEMA_VERSION = 2
DEMO_VERIFICATION_STATUS = "DEMO · NON-CONFIRMATORY · PAIRED TAPE"

AGENT_PRESENTATION: Mapping[str, tuple[str, str, str]] = {
    "planning": (
        "Open-loop A*",
        "Planning",
        "Plans once against the nominal map and deliberately does not repair disruptions.",
    ),
    "replanning": (
        "Event-triggered A*",
        "Replanning",
        "Repairs the current route when closures, arrivals, or service state invalidate it.",
    ),
    "q-learning": (
        "Q-learning",
        "Model-free RL",
        "Learns primitive action values from real interactions over an exact tabular state.",
    ),
    "dyna-q": (
        "Dyna-Q",
        "Model-based RL",
        "Adds seeded simulated updates from a learned deterministic transition model.",
    ),
    "dqn": (
        "Goal-guided DQN",
        "Deep RL",
        "Combines learned primitive action values with a transparent goal-direction prior.",
    ),
    "hybrid": (
        "Learning + A*",
        "Hybrid",
        "Learns the next task option while A* supplies inspectable, constraint-aware routing.",
    ),
}

AGENT_COLORS: Mapping[str, str] = {
    "planning": "#727776",
    "replanning": "#1d6f68",
    "q-learning": "#c3983f",
    "dyna-q": "#8b6fb0",
    "dqn": "#397caf",
    "hybrid": "#dc735b",
}


@dataclass(frozen=True)
class _GalleryCaseSpec:
    case_id: str
    map_id: str
    label: str
    description: str
    tags: list[str]
    display: dict[str, object]
    scenario: Scenario
    training_episodes: Mapping[str, int]


def make_demo_agents() -> tuple[Agent, ...]:
    """Construct fresh agents with small, documented demonstration settings."""

    return (
        OpenLoopPlanningAgent(),
        ReplanningAgent(),
        QLearningAgent(
            alpha=0.15,
            epsilon_decay=0.999,
            epsilon_min=0.05,
            guidance_probability=0.85,
            progress_reward=0.12,
        ),
        DynaQAgent(
            alpha=0.15,
            epsilon_decay=0.998,
            epsilon_min=0.05,
            planning_steps=8,
            guidance_probability=0.75,
            progress_reward=0.12,
        ),
        DQNAgent(
            DQNConfig(
                hidden_sizes=(64, 64),
                batch_size=32,
                replay_capacity=5_000,
                warmup_steps=64,
                update_every=4,
                target_sync_interval=100,
                epsilon_decay_steps=5_000,
                guidance_probability=0.75,
                guidance_bias=8.0,
                progress_reward=0.12,
            )
        ),
        HybridAgent(HybridConfig(hidden=(32, 32), epsilon=0.20)),
    )


def build_demo_data(
    scenario: Scenario,
    *,
    root_seed: int = 42,
    training_episodes: Mapping[str, int] = DEFAULT_TRAINING_EPISODES,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Train configured learners, then evaluate every method on one paired tape."""

    results: list[EpisodeResult] = []
    training_counts: dict[str, int] = {}
    for index, agent in enumerate(make_demo_agents()):
        episodes = int(training_episodes.get(agent.name, 0)) if agent.trainable else 0
        if episodes < 0:
            raise ValueError("training episode counts must be non-negative")
        training_counts[agent.name] = episodes
        if episodes:
            train_episodes(
                agent,
                (scenario,),
                episodes=episodes,
                root_seed=root_seed * 10_000 + index * 1_000,
            )
        results.append(
            run_episode(
                agent,
                scenario,
                seed=root_seed,
                explore=False,
                learn=False,
                measure_timing=False,
            )
        )

    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    return {
        "generatedAt": timestamp,
        "rootSeed": root_seed,
        "verificationStatus": DEMO_VERIFICATION_STATUS,
        "scenarioFingerprint": fingerprint(scenario.to_dict()),
        "scenario": _scenario_payload(scenario),
        "trainingEpisodes": training_counts,
        "agents": [_agent_payload(result) for result in results],
    }


def write_demo_data(
    path: Path,
    scenario: Scenario,
    *,
    root_seed: int = 42,
    training_episodes: Mapping[str, int] = DEFAULT_TRAINING_EPISODES,
) -> dict[str, object]:
    payload = build_demo_data(
        scenario,
        root_seed=root_seed,
        training_episodes=training_episodes,
    )
    write_json_atomic(path, payload)
    return payload


def build_demo_gallery(
    config_path: str | Path,
    *,
    root_seed: int = 42,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Build a versioned gallery from scenarios declared relative to one config file.

    Every case delegates to :func:`build_demo_data`, which constructs a fresh set
    of agents. Learning state is therefore never carried from one gallery case to
    the next.
    """

    resolved_config = Path(config_path).resolve()
    raw_config: object = json.loads(resolved_config.read_text(encoding="utf-8"))
    config = _expect_mapping(raw_config, "gallery config")
    default_case_id = _expect_string(
        _required(config, "defaultCaseId", "gallery config"),
        "gallery config.defaultCaseId",
    )
    raw_cases = _required(config, "cases", "gallery config")
    if not isinstance(raw_cases, list):
        raise TypeError("gallery config.cases must be a JSON array")
    if not raw_cases:
        raise ValueError("gallery config.cases must not be empty")

    training_episodes = _training_episode_config(
        config.get("trainingEpisodes", DEFAULT_TRAINING_EPISODES),
        "gallery config.trainingEpisodes",
    )
    timestamp = generated_at or datetime.now(UTC)
    validated_cases: list[_GalleryCaseSpec] = []
    case_ids: set[str] = set()
    map_ids: set[str] = set()

    for index, raw_case in enumerate(raw_cases):
        path = f"gallery config.cases[{index}]"
        case = _expect_mapping(raw_case, path)
        case_id = _expect_string(_required(case, "caseId", path), f"{path}.caseId")
        map_id = _expect_string(_required(case, "mapId", path), f"{path}.mapId")
        if case_id in case_ids:
            raise ValueError(f"gallery case IDs must be unique: {case_id!r}")
        if map_id in map_ids:
            raise ValueError(f"gallery map IDs must be unique: {map_id!r}")
        case_ids.add(case_id)
        map_ids.add(map_id)

        label = _expect_string(_required(case, "label", path), f"{path}.label")
        description = _expect_string(
            _required(case, "description", path), f"{path}.description"
        )
        tags = _expect_string_list(_required(case, "tags", path), f"{path}.tags")
        display = _expect_mapping(_required(case, "display", path), f"{path}.display")
        topology = _expect_string(
            _required(display, "topology", f"{path}.display"),
            f"{path}.display.topology",
        )
        difficulty = _expect_string(
            _required(display, "difficulty", f"{path}.display"),
            f"{path}.display.difficulty",
        )
        display_payload = dict(display)
        display_payload.update(topology=topology, difficulty=difficulty)

        scenario_reference = _expect_string(
            _required(case, "scenario", path), f"{path}.scenario"
        )
        scenario_path = Path(scenario_reference)
        if not scenario_path.is_absolute():
            scenario_path = resolved_config.parent / scenario_path
        scenario = Scenario.from_json(scenario_path.read_text(encoding="utf-8"))
        case_training_episodes = _training_episode_config(
            case.get("trainingEpisodes", training_episodes),
            f"{path}.trainingEpisodes",
        )

        validated_cases.append(
            _GalleryCaseSpec(
                case_id=case_id,
                map_id=map_id,
                label=label,
                description=description,
                tags=tags,
                display=display_payload,
                scenario=scenario,
                training_episodes=case_training_episodes,
            )
        )

    if default_case_id not in case_ids:
        raise ValueError(
            "gallery config.defaultCaseId must match one configured case ID: "
            f"{default_case_id!r}"
        )

    cases: list[dict[str, object]] = []
    for case_spec in validated_cases:
        demo = build_demo_data(
            case_spec.scenario,
            root_seed=root_seed,
            training_episodes=case_spec.training_episodes,
            generated_at=timestamp,
        )
        cases.append(
            {
                "caseId": case_spec.case_id,
                "mapId": case_spec.map_id,
                "label": case_spec.label,
                "description": case_spec.description,
                "tags": case_spec.tags,
                "display": case_spec.display,
                "scenarioFingerprint": demo["scenarioFingerprint"],
                "scenario": demo["scenario"],
                "trainingEpisodes": demo["trainingEpisodes"],
                "agents": demo["agents"],
            }
        )

    return {
        "schemaVersion": DEMO_GALLERY_SCHEMA_VERSION,
        "generatedAt": timestamp.astimezone(UTC).isoformat(),
        "rootSeed": root_seed,
        "verificationStatus": DEMO_VERIFICATION_STATUS,
        "defaultCaseId": default_case_id,
        "cases": cases,
    }


def write_demo_gallery(
    path: Path,
    config_path: str | Path,
    *,
    root_seed: int = 42,
) -> dict[str, object]:
    """Build and atomically write a version-two demo gallery artifact."""

    payload = build_demo_gallery(config_path, root_seed=root_seed)
    write_json_atomic(path, payload)
    return payload


def _scenario_payload(scenario: Scenario) -> dict[str, object]:
    return {
        "id": scenario.scenario_id,
        "width": scenario.map.width,
        "height": scenario.map.height,
        "horizon": scenario.horizon,
        "batteryCapacity": scenario.battery_capacity,
        "initialRobot": {
            "x": scenario.initial_robot.position.x,
            "y": scenario.initial_robot.position.y,
        },
        "obstacles": [
            {"x": position.x, "y": position.y}
            for position in sorted(scenario.map.obstacles)
        ],
        "chargingStations": [
            {"x": position.x, "y": position.y}
            for position in sorted(scenario.map.charging_stations)
        ],
        "orders": [
            {
                "id": order.order_id,
                "pickup": {"x": order.pickup.x, "y": order.pickup.y},
                "dropoff": {"x": order.dropoff.x, "y": order.dropoff.y},
                "releaseTime": order.release_time,
                "deadline": order.deadline,
                "priority": order.priority,
            }
            for order in scenario.orders
        ],
        "events": [
            {
                "time": event.time,
                "kind": event.kind.value,
                **(
                    {"position": {"x": event.position.x, "y": event.position.y}}
                    if event.position is not None
                    else {}
                ),
                **({"orderId": event.order_id} if event.order_id is not None else {}),
            }
            for event in scenario.event_tape
        ],
    }


def _agent_payload(result: EpisodeResult) -> dict[str, object]:
    label, family, description = AGENT_PRESENTATION[result.agent]
    metrics = result.metrics
    return {
        "id": result.agent,
        "label": label,
        "family": family,
        "description": description,
        "color": AGENT_COLORS[result.agent],
        "metrics": {
            "weightedCompletionRate": metrics.weighted_completion_rate,
            "weightedOnTimeCompletionRate": metrics.weighted_on_time_completion_rate,
            "totalReward": metrics.total_reward,
            "completedOrders": metrics.completed_orders,
            "totalOrders": metrics.total_orders,
            "constraintViolations": metrics.constraint_violations,
            # Timing is deliberately disabled for deterministic, portable demo tapes.
            "decisionTimeMs": None,
            "steps": metrics.steps,
        },
        "planningCalls": result.diagnostics.planning_calls,
        "expandedNodes": result.diagnostics.expanded_nodes,
        "learningUpdates": result.diagnostics.learning_updates,
        "trace": [
            {
                "time": record.time,
                "action": record.action,
                "position": list(record.position),
                "battery": record.battery,
                "reward": record.reward,
                "cumulativeReward": record.cumulative_reward,
                "carriedOrderId": record.carried_order_id,
                "deliveredOrderId": record.delivered_order_id,
                "violations": list(record.violations),
                "eventCount": record.event_count,
            }
            for record in result.trace
        ],
    }


def _required(data: Mapping[str, object], key: str, path: str) -> object:
    try:
        return data[key]
    except KeyError as error:
        raise ValueError(f"{path} is missing required field {key!r}") from error


def _expect_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings")
    return cast(Mapping[str, object], value)


def _expect_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{path} must not be empty")
    return normalized


def _expect_string_list(value: object, path: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array")
    result = [_expect_string(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise ValueError(f"{path} must not contain duplicates")
    return result


def _training_episode_config(value: object, path: str) -> Mapping[str, int]:
    data = _expect_mapping(value, path)
    unknown = sorted(set(data) - set(DEFAULT_TRAINING_EPISODES))
    if unknown:
        raise ValueError(f"{path} contains unknown agents: {unknown!r}")

    counts: dict[str, int] = {}
    for agent, episodes in data.items():
        if isinstance(episodes, bool) or not isinstance(episodes, int):
            raise TypeError(f"{path}.{agent} must be an integer")
        if episodes < 0:
            raise ValueError(f"{path}.{agent} must be non-negative")
        counts[agent] = episodes
    return counts


__all__ = [
    "DEFAULT_TRAINING_EPISODES",
    "DEMO_GALLERY_SCHEMA_VERSION",
    "DEMO_VERIFICATION_STATUS",
    "build_demo_data",
    "build_demo_gallery",
    "make_demo_agents",
    "write_demo_data",
    "write_demo_gallery",
]

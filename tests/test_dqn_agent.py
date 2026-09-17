from __future__ import annotations

import unittest
from copy import deepcopy

import numpy as np

from adaptive_agent_lab.agents.dqn import (
    DQNAgent,
    DQNConfig,
    config_from_mapping,
)
from adaptive_agent_lab.environment.contracts import (
    Action,
    Order,
    Position,
    RobotState,
    WarehouseMap,
)
from adaptive_agent_lab.environment.events import EventTape
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.environment.simulator import WarehouseEnvironment


def tiny_environment() -> WarehouseEnvironment:
    scenario = Scenario(
        scenario_id="dqn-test",
        warehouse_map=WarehouseMap(
            3,
            2,
            obstacles=frozenset({Position(0, 1)}),
            charging_stations=frozenset({Position(2, 1)}),
        ),
        orders=(Order("a", Position(1, 0), Position(2, 0), 0, 8),),
        initial_robot=RobotState(Position(0, 0), 6),
        event_tape=EventTape(),
        horizon=8,
        battery_capacity=6,
    )
    return WarehouseEnvironment(scenario)


def one_step(agent: DQNAgent, environment: WarehouseEnvironment, *, explore: bool) -> Action:
    action = agent.act(environment.snapshot, explore=explore)
    environment.step(action)
    agent.observe(environment.history[-1])
    return action


class DQNAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DQNConfig(
            hidden_sizes=(8,),
            batch_size=1,
            replay_capacity=10,
            warmup_steps=1,
            target_sync_interval=1,
            epsilon_start=1.0,
            epsilon_end=0.0,
            epsilon_decay_steps=10,
        )

    def test_one_transition_runs_a_learning_update(self) -> None:
        environment = tiny_environment()
        agent = DQNAgent(self.config)
        agent.reset(environment.snapshot, seed=42)
        one_step(agent, environment, explore=True)
        self.assertEqual(agent.replay_size, 1)
        self.assertEqual(agent.update_count, 1)
        self.assertIsNotNone(agent.last_loss)

    def test_selected_action_always_respects_the_shared_mask(self) -> None:
        environment = tiny_environment()
        agent = DQNAgent(self.config)
        agent.reset(environment.snapshot, seed=3)
        actions = {agent.act(environment.snapshot, explore=True) for _ in range(50)}
        self.assertNotIn(Action.LEFT, actions)
        self.assertNotIn(Action.UP, actions)
        self.assertNotIn(Action.DOWN, actions)
        self.assertNotIn(Action.DROPOFF, actions)
        self.assertNotIn(Action.CHARGE, actions)

    def test_same_seed_produces_same_exploration_stream(self) -> None:
        first_environment = tiny_environment()
        second_environment = tiny_environment()
        first = DQNAgent(self.config)
        second = DQNAgent(self.config)
        first.reset(first_environment.snapshot, seed=99)
        second.reset(second_environment.snapshot, seed=99)
        first_actions: list[Action] = []
        second_actions: list[Action] = []
        for _ in range(4):
            first_actions.append(one_step(first, first_environment, explore=True))
            second_actions.append(one_step(second, second_environment, explore=True))
        self.assertEqual(first_actions, second_actions)

    def test_model_state_round_trip_preserves_q_values(self) -> None:
        environment = tiny_environment()
        original = DQNAgent(self.config)
        original.reset(environment.snapshot, seed=8)
        one_step(original, environment, explore=True)
        expected = original.q_values(environment.snapshot)

        restored = DQNAgent(self.config)
        restored.load_state_dict(original.state_dict())
        restored.reset(environment.snapshot, seed=999)
        self.assertTrue(np.allclose(expected, restored.q_values(environment.snapshot)))

    def test_evaluation_transition_does_not_change_learning_state(self) -> None:
        environment = tiny_environment()
        agent = DQNAgent(self.config)
        agent.reset(environment.snapshot, seed=8)
        one_step(agent, environment, explore=True)
        before = deepcopy(agent.state_dict())
        replay_size = agent.replay_size

        agent.set_learning_enabled(False)
        one_step(agent, environment, explore=False)

        self.assertEqual(agent.state_dict(), before)
        self.assertEqual(agent.replay_size, replay_size)
        self.assertEqual(agent.diagnostics.learning_updates, 1)

    def test_observe_requires_the_action_selected_by_act(self) -> None:
        environment = tiny_environment()
        agent = DQNAgent(self.config)
        agent.reset(environment.snapshot, seed=2)
        environment.step(Action.WAIT)
        with self.assertRaises(RuntimeError):
            agent.observe(environment.history[-1])

        selected = agent.act(environment.snapshot)
        wrong = Action.WAIT if selected is not Action.WAIT else Action.RIGHT
        environment.step(wrong)
        with self.assertRaises(ValueError):
            agent.observe(environment.history[-1])

    def test_reset_rejects_incompatible_observation_shape(self) -> None:
        environment = tiny_environment()
        agent = DQNAgent(self.config)
        agent.reset(environment.snapshot, seed=1)
        incompatible = WarehouseEnvironment(
            Scenario(
                scenario_id="dqn-incompatible",
                warehouse_map=WarehouseMap(4, 2),
                orders=(Order("a", Position(1, 0), Position(3, 0), 0, 8),),
                initial_robot=RobotState(Position(0, 0), 6),
                event_tape=EventTape(),
                horizon=8,
                battery_capacity=6,
            )
        )
        with self.assertRaises(ValueError):
            agent.reset(incompatible.snapshot, seed=2)

    def test_clear_learning_discards_network_replay_and_counters(self) -> None:
        environment = tiny_environment()
        agent = DQNAgent(self.config)
        agent.reset(environment.snapshot, seed=9)
        one_step(agent, environment, explore=True)
        agent.clear_learning()

        self.assertEqual(agent.replay_size, 0)
        self.assertEqual(agent.update_count, 0)
        self.assertIsNone(agent.last_loss)
        with self.assertRaises(RuntimeError):
            agent.q_values(environment.snapshot)

        agent.reset(environment.snapshot, seed=10)
        self.assertEqual(agent.q_values(environment.snapshot).shape, (len(Action),))

    def test_state_loader_rejects_unsupported_or_inconsistent_states(self) -> None:
        environment = tiny_environment()
        original = DQNAgent(self.config)
        original.reset(environment.snapshot, seed=4)
        valid = original.state_dict()

        with self.assertRaises(ValueError):
            DQNAgent(self.config).load_state_dict({"format_version": 2, "agent": "dqn"})
        with self.assertRaises(ValueError):
            DQNAgent(self.config).load_state_dict(
                {"format_version": 1, "agent": "dqn", "network": [], "target": {}}
            )

        different = DQNAgent(DQNConfig(hidden_sizes=(4,)))
        different.reset(environment.snapshot, seed=4)
        mismatched = deepcopy(valid)
        mismatched["target"] = different.state_dict()["target"]
        with self.assertRaises(ValueError):
            DQNAgent(self.config).load_state_dict(mismatched)

        for key, invalid in (("environment_steps", -1), ("updates", True)):
            malformed = deepcopy(valid)
            malformed[key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                DQNAgent(self.config).load_state_dict(malformed)


class DQNConfigTests(unittest.TestCase):
    def test_config_validates_all_training_bounds(self) -> None:
        invalid_arguments = (
            {"hidden_sizes": ()},
            {"hidden_sizes": (8, 8, 8)},
            {"hidden_sizes": (0,)},
            {"gamma": -0.1},
            {"gamma": 1.1},
            {"learning_rate": 0.0},
            {"max_grad_norm": 0.0},
            {"batch_size": 0},
            {"replay_capacity": 0},
            {"update_every": 0},
            {"target_sync_interval": 0},
            {"epsilon_decay_steps": 0},
            {"warmup_steps": -1},
            {"epsilon_start": 1.1},
            {"epsilon_end": -0.1},
            {"epsilon_start": 0.2, "epsilon_end": 0.3},
            {"guidance_bias": -0.1},
            {"guidance_probability": 1.1},
            {"progress_reward": -0.1},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                DQNConfig(**arguments)

    def test_config_mapping_supports_json_values_and_rejects_string_hidden_layers(self) -> None:
        config = config_from_mapping(
            {
                "hidden_sizes": [16],
                "gamma": "0.8",
                "batch_size": "4",
                "warmup_steps": "0",
            }
        )
        self.assertEqual(config.hidden_sizes, (16,))
        self.assertEqual(config.gamma, 0.8)
        self.assertEqual(config.batch_size, 4)
        with self.assertRaises(TypeError):
            config_from_mapping({"hidden_sizes": "16,16"})

    def test_constructor_requires_a_dqn_config(self) -> None:
        with self.assertRaises(TypeError):
            DQNAgent({})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

# Adaptive Agent Lab

[![CI](https://github.com/Revincxt/adaptive-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Revincxt/adaptive-agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![Live demo](https://img.shields.io/badge/Live_demo-GitHub_Pages-2F81F7.svg?logo=github)](https://revincxt.github.io/adaptive-agent/)

A reproducible Python lab for comparing planning, reinforcement-learning, and
hybrid agents in a deterministic, dynamic warehouse simulator.

**[Open the replay explorer](https://revincxt.github.io/adaptive-agent/)** to
inspect six controllers across four warehouse layouts.

> **Status: alpha.** The simulator, agents, benchmark runner, artifacts, and
> replay exporter are implemented and tested. Checkpoint-aware confirmatory
> evaluation is not implemented; committed replay data is demonstration-only
> and must not be presented as an algorithm ranking.

## Agents

| ID | Implementation |
| --- | --- |
| `planning` | Open-loop A* |
| `replanning` | State-triggered A* replanning |
| `q-learning` | Exact tabular Q-learning |
| `dyna-q` | Q-learning with a learned one-step model |
| `dqn` | Goal-guided NumPy DQN |
| `hybrid` | Learned high-level options with A* routing |

All agents use the same actions, simulator, seeded scenarios, and artifact
contracts. See [problem formulation](docs/problem-formulation.md),
[architecture](docs/architecture.md), and
[experiment protocol](docs/experiment-protocol.md) for details.

## Install

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Use

Inspect or generate a scenario:

```bash
aal scenario show scenarios/small/dynamic-demo.json
aal scenario generate scenarios/small/example.json \
  --size small --dynamics medium --orders 4 --horizon 180 --seed 42
```

Run, train, or benchmark an agent:

```bash
aal run --agent replanning \
  --scenario scenarios/small/dynamic-demo.json --seed 42

aal train --agent dyna-q \
  --scenario scenarios/small/dynamic-demo.json \
  --episodes 100 --output models/dyna-q-demo.json

aal benchmark --config configs/benchmarks/main.json \
  --agents planning,replanning --quick 2 --output runs/smoke
```

Run `aal --help` for all commands and options.

## Replay explorer

Regenerate the committed gallery:

```bash
aal export-gallery --config configs/demo-gallery.json \
  --output web/public/demo-data.json --seed 42
```

Run it locally with Node.js 22.13+ and pnpm:

```bash
cd web
pnpm install
pnpm dev
```

## Verify

```bash
python -m pytest
python -m ruff check .
python -m mypy src
```

The current benchmark command creates fresh agents. Learning-agent runs without
loaded checkpoints are **untrained smoke tests**, not research results. Cite the
generated manifest and summary for any reported run.

## Scope

Version 0.2 models one fully observable robot with discrete time, deterministic
primitive actions, one carried order, dynamic order arrivals, and temporary
cell blockages. Multi-robot coordination, partial observability, continuous
control, and real warehouse integrations are out of scope.

Licensed under the [MIT License](LICENSE).

# Problem formulation

## Material Passport

- Origin skill: `experiment-agent`
- Origin mode: `plan`
- Origin date: 2026-08-05
- Verification status: **IMPLEMENTATION-ALIGNED; EMPIRICAL RESULTS UNVERIFIED**
- Version label: `problem-formulation-v0.1`

This document specifies the environment implemented by Adaptive Agent Lab. It
defines the information boundary, action semantics, transition order, reward,
and operational metrics. It does not report comparative results.

## Research task

One robot performs pickup-and-delivery work in a fully observable,
discrete-time warehouse. Orders have release times, active cells can become
temporarily blocked and later reopen, and the robot has finite battery. Given a
scenario and action sequence, environment transitions are deterministic.

Version 0.1 excludes multiple robots, hidden order records, stochastic action
effects, continuous control, and real warehouse integration.

## State and fixed context

A scenario fixes the map, complete order list, initial robot, event tape,
horizon `H`, and battery capacity. Runtime state is

```text
s_t = (t, robot_position, battery, carried_order_id,
       order_statuses, blocked_cells, cumulative_reward, terminated)
```

The immutable map and order definitions remain available through every
`WarehouseSnapshot`. The event tape itself is not exposed to an agent.

### Coordinates and map

Coordinates are `(x, y)` with `(0, 0)` at the top-left. `x` increases to the
right and `y` increases downward. Movement is four-connected.

The map records its width, height, permanent obstacles, and charging stations.
Pickup and drop-off cells belong to orders rather than to separate map layers.
The robot always occupies a statically traversable cell.

### Robot and battery

The robot carries at most one order. A successful movement consumes exactly one
battery unit. `charge` adds the environment's charge rate (2 by default),
capped at battery capacity. Service actions and `wait` do not consume battery.
Every attempted action advances one tick while the episode is active.

### Orders

Each immutable order contains

```text
(order_id, pickup, dropoff, release_time, deadline, priority)
```

`priority` is a positive finite weight. Runtime status is one of `pending`,
`available`, `picked_up`, `delivered`, or `expired`.

All order definitions—including pending orders and their future release
times—are present in the common snapshot. `order_arrival` changes a pending
order to available; it does not reveal a previously hidden record. This is an
intentional full-information task definition. Future cell blockage events are
not exposed.

At the horizon, every non-terminal order becomes `expired`. Missing a deadline
before the horizon does not itself expire an order; a late delivery can still
occur.

### Temporary blockages

`cell_blocked` prevents entry into a cell until `cell_unblocked`. A cell may be
blocked while the robot occupies it; the robot may leave but cannot re-enter
until it reopens. The bundled generator protects the initial robot cell,
charging stations, pickups, and drop-offs from generated blockages, although
the scenario schema permits a manually authored event on any statically
traversable cell.

## Observation representations

Every agent receives the same immutable `WarehouseSnapshot`. It contains the
complete map, current runtime state, complete order tuple, horizon, and battery
capacity. It contains no future `EventTape` entries and no other agent's state.

Three derived views are shared across implementations.

### Neural vector

The DQN and hybrid agents use a padded flat vector of length

```text
4 * (width * height) + 6 + 12 * max_orders
```

The four grid channels are:

1. permanent obstacles;
2. charging stations;
3. current dynamic blockages; and
4. current robot position.

They are flattened in row-major cell order. The six global features are
normalized time, normalized battery, a carrying flag, normalized relative
`x`/`y` offsets to the active service target, and a target-present flag. The
target is the carried order's drop-off or the highest-priority available
pickup under the shared deadline ordering. Each order slot has:

- normalized pickup `x` and `y`;
- normalized drop-off `x` and `y`;
- normalized release time and deadline;
- normalized priority; and
- one-hot status across the five order states.

Unused slots are zero padded. There are no pickup, drop-off, priority, or
urgency grid channels.

### Exact tabular state

Q-learning and Dyna-Q use

```text
(time,
 flattened_robot_cell,
 battery,
 carried_order_index,
 one_status_code_per_order,
 dynamic_blockage_bitmask)
```

This is an exact representation of mutable runtime state given the fixed map
and order context. It does not use quartiles, relative directions, distance
bins, deadline bins, local-only closure masks, or order-count bins.

### Immediate action mask

The common encoder marks immediately feasible actions. Learning agents use this
mask for greedy and exploratory selection, while the simulator independently
validates every action. The demo DQN additionally applies a documented bias to
the goal-directed feasible action; this is why it is labelled goal-guided DQN.
`wait` is always masked in for a live episode. `charge` is masked in only at a
charging station when battery is below capacity.

## Action space

The serialized action names and enumeration order are:

```text
up, down, left, right, pickup, dropoff, charge, wait
```

| Action | Simulator precondition | Successful effect |
| --- | --- | --- |
| `up` | positive battery; `(x, y-1)` is in bounds and unblocked | move and spend one battery |
| `down` | positive battery; `(x, y+1)` is in bounds and unblocked | move and spend one battery |
| `left` | positive battery; `(x-1, y)` is in bounds and unblocked | move and spend one battery |
| `right` | positive battery; `(x+1, y)` is in bounds and unblocked | move and spend one battery |
| `pickup` | robot carries nothing and an available order has this pickup cell | set one order to `picked_up` and carry it |
| `dropoff` | robot carries an order and is at that order's drop-off cell | set it to `delivered` and clear the carried ID |
| `charge` | robot is at a charging station | add charge up to capacity |
| `wait` | episode is active | no service, movement, or battery change |

If multiple available orders share a pickup cell, the environment chooses
earliest deadline, then highest priority, then lexicographically smallest order
ID.

An invalid action is retained as the recorded action. Its movement or service
effect is a no-op; time still advances, a typed violation is recorded, and the
invalid-action cost is added. It is therefore behaviorally wait-like for robot
position but is not rewritten to the `wait` action.

## Transition ordering

Reset constructs the initial statuses and applies every event at time 0. The
agent then receives the time-0 snapshot. For each live step:

1. the agent selects one action from the snapshot at time `t`;
2. the simulator applies or rejects that action and accumulates its reward;
3. the clock advances to `t + 1`;
4. all events scheduled at `t + 1` are applied in canonical order;
5. completion and horizon termination are evaluated; and
6. the next immutable state and transition are published.

An episode ends as soon as all orders are delivered, or at `H`, when unfinished
orders become expired. Episodes are not forced to run for exactly `H` steps.

## EventTape contract

Supported event kinds are `order_arrival`, `cell_blocked`, and
`cell_unblocked`. Events are sorted by the total key

```text
(time, kind, order_id, position.x, position.y)
```

The tape rejects duplicate events and two events for the same target at the same
time. Every non-zero order release time must have exactly one matching arrival
event. A cell must be blocked before it can be unblocked.

Scenario generation derives separate stable seeds for geometry, orders, and
events. The paired benchmark derives scenario seeds without an agent label and
records a fingerprint of the complete scenario, which includes the event tape.

## Reward

The default `RewardScheme` used by `WarehouseEnvironment` is:

| Term | Implemented value | When applied |
| --- | ---: | --- |
| `step_cost` | `-0.05` | every live step |
| `movement_cost` | `-0.02` | each successful movement |
| `invalid_action_cost` | `-1.0` | an invalid attempted action |
| `pickup_reward` | `+0.25` | successful pickup |
| `delivery_reward_per_priority` | `+10.0` | successful delivery, multiplied by priority |
| `on_time_bonus_per_priority` | `+5.0` | delivery at or before deadline, multiplied by priority |
| `lateness_cost_per_step` | `-0.20` | late delivery, multiplied by lateness ticks only |
| `stranded_cost` | `-10.0` | declared in the scheme but not applied by the v0.1 simulator |

For a delivery completed at time `c` with deadline `d` and priority `p`:

```text
delivery(c, d, p) = 10p + 5p                    if c <= d
                    10p - 0.20 * (c - d)       if c > d
```

There is no per-tick active-tardiness penalty and no unfinished-order horizon
penalty in the current simulator. Episode return is a learning diagnostic, not
the primary cross-family outcome.

## Evaluation outcomes

The primary planned outcome is weighted on-time completion rate (WOTCR):

```text
sum(priority of orders delivered at or before deadline)
-------------------------------------------------------
sum(priority of every order in the scenario)
```

The episode runner also records total reward, weighted completion rate,
completed and total order counts, mean lateness among delivered orders, valid
movement steps, valid movement steps per completed order, constraint-violation
count, aggregate decision time, and episode steps. Agent diagnostics add A*
planning calls, expanded nodes, and learning updates.

Metric aliases in benchmark reports include `total_task_utility`,
`mean_tardiness_ticks`, `movement_steps`, `energy_used`,
`astar_nodes_expanded`, and `replan_count`. In v0.1, `energy_used` aliases valid
movement steps because each successful move consumes one unit.

## Agent treatments

| ID | Implemented decision rule |
| --- | --- |
| `planning` | Plan once over orders available at reset. Later arrivals are ignored; blocked planned moves can become invalid attempts. |
| `replanning` | Rebuild a one-goal A* route when blockages, statuses, carried order, or next-action feasibility changes. |
| `q-learning` | Masked primitive-action Q-learning over the exact tabular state. |
| `dyna-q` | The same Q update plus seeded model updates from the latest observed state-action transition. |
| `dqn` | Masked standard DQN over the flat padded neural vector. |
| `hybrid` | Learn per-order/charge/wait options and execute active goals through blockage-aware A*. |

A* uses Manhattan distance and the current blockage snapshot only. Its movement
expansion order is `up`, `down`, `left`, `right`, with stable insertion-order
ties.

## Interpretation boundary

The environment is a deterministic research simulator. Its hand-designed
reward, complete knowledge of pending orders, generated layouts, and hard action
masks give all conclusions a narrow scope. Demonstration and untrained smoke
runs are not evidence of comparative algorithm quality. Any change to state,
event, reward, metric, or information semantics requires a versioned protocol
update before a confirmatory benchmark.

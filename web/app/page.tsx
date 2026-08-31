"use client";

import { useEffect, useState, type CSSProperties } from "react";

type Point = { x: number; y: number };
type Order = {
  id: string;
  pickup: Point;
  dropoff: Point;
  releaseTime: number;
  deadline: number;
  priority: number;
};
type Event = {
  time: number;
  kind: "order_arrival" | "cell_blocked" | "cell_unblocked";
  position?: Point;
  orderId?: string;
};
type TraceStep = {
  time: number;
  action: string;
  position: [number, number];
  battery: number;
  reward: number;
  cumulativeReward: number;
  carriedOrderId: string | null;
  deliveredOrderId: string | null;
  violations: string[];
  eventCount: number;
};
type Metrics = {
  weightedCompletionRate: number;
  weightedOnTimeCompletionRate: number;
  totalReward: number;
  completedOrders: number;
  totalOrders: number;
  constraintViolations: number;
  decisionTimeMs: number | null;
  steps: number;
};
type AgentResult = {
  id: string;
  label: string;
  family: string;
  description: string;
  color: string;
  metrics: Metrics;
  planningCalls: number;
  expandedNodes: number;
  learningUpdates: number;
  trace: TraceStep[];
};
type ScenarioData = {
  id: string;
  width: number;
  height: number;
  horizon: number;
  batteryCapacity: number;
  initialRobot: Point;
  obstacles: Point[];
  chargingStations: Point[];
  orders: Order[];
  events: Event[];
};
type DemoCase = {
  caseId: string;
  mapId: string;
  label: string;
  description: string;
  tags: string[];
  display?: {
    topology?: string;
    difficulty?: string;
    [key: string]: unknown;
  };
  scenarioFingerprint: string;
  scenario: ScenarioData;
  trainingEpisodes: Record<string, number>;
  agents: AgentResult[];
};
type DemoBundle = {
  schemaVersion: number;
  generatedAt: string;
  rootSeed: number;
  verificationStatus: string;
  defaultCaseId: string;
  cases: DemoCase[];
};
type RouteDirection = "north" | "east" | "south" | "west";
type RoutePhase = "primary" | "recorded" | "reference";
type OrderState = "queued" | "ready" | "carried" | "delivered" | "expired";

const actionLabels: Record<string, string> = {
  up: "Move north",
  down: "Move south",
  left: "Move west",
  right: "Move east",
  pickup: "Pick up order",
  dropoff: "Deliver order",
  charge: "Recharge",
  wait: "Wait",
};

const eventLabels: Record<Event["kind"], string> = {
  order_arrival: "Order released",
  cell_blocked: "Aisle closed",
  cell_unblocked: "Aisle reopened",
};

const playbackRates = [0.5, 1, 2] as const;
const routeDisplayColors: Record<string, string> = {
  planning: "#4f5656",
  replanning: "#125a55",
  "q-learning": "#765511",
  "dyna-q": "#674a88",
  dqn: "#275f8d",
  hybrid: "#a34734",
};
const cellKey = (point: Point) => `${point.x}:${point.y}`;

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en", { maximumFractionDigits: 1 }).format(value);
}

function formatSigned(value: number) {
  const rounded = Number(value.toFixed(1));
  return rounded > 0 ? `+${rounded}` : `${rounded}`;
}

function pointFromTrace(step: TraceStep): Point {
  return { x: step.position[0], y: step.position[1] };
}

function endTime(agent: AgentResult | null) {
  return agent?.trace.at(-1)?.time ?? 1;
}

function replayEndTime(
  scenario: ScenarioData,
  primary: AgentResult | null,
  reference: AgentResult | null,
) {
  const latestEvent = Math.max(1, ...scenario.events.map((event) => event.time));
  return Math.min(scenario.horizon, Math.max(endTime(primary), endTime(reference), latestEvent));
}

function stepAtTime(agent: AgentResult, time: number) {
  let current = agent.trace[0];
  for (const step of agent.trace) {
    if (step.time > time) break;
    current = step;
  }
  return current;
}

function blockedCells(events: Event[], time: number) {
  const cells = new Set<string>();
  for (const event of events) {
    if (event.time > time || !event.position) continue;
    const key = cellKey(event.position);
    if (event.kind === "cell_blocked") cells.add(key);
    if (event.kind === "cell_unblocked") cells.delete(key);
  }
  return cells;
}

function buildRoute(points: Point[]) {
  const cells = new Map<string, Set<RouteDirection>>();
  const ensureCell = (point: Point) => {
    const key = cellKey(point);
    if (!cells.has(key)) cells.set(key, new Set<RouteDirection>());
    return cells.get(key)!;
  };

  points.forEach((point) => ensureCell(point));
  for (let index = 1; index < points.length; index += 1) {
    const from = points[index - 1];
    const to = points[index];
    const fromCell = ensureCell(from);
    const toCell = ensureCell(to);

    if (to.x === from.x + 1 && to.y === from.y) {
      fromCell.add("east");
      toCell.add("west");
    } else if (to.x === from.x - 1 && to.y === from.y) {
      fromCell.add("west");
      toCell.add("east");
    } else if (to.y === from.y + 1 && to.x === from.x) {
      fromCell.add("south");
      toCell.add("north");
    } else if (to.y === from.y - 1 && to.x === from.x) {
      fromCell.add("north");
      toCell.add("south");
    }
  }
  return cells;
}

function RouteLayer({
  points,
  width,
  height,
  phase,
}: {
  points: Point[];
  width: number;
  height: number;
  phase: RoutePhase;
}) {
  const route = buildRoute(points);
  return (
    <div
      className={`route-layer route-${phase}`}
      style={{
        gridTemplateColumns: `repeat(${width}, 1fr)`,
        gridTemplateRows: `repeat(${height}, 1fr)`,
      }}
      aria-hidden="true"
    >
      {Array.from({ length: width * height }).map((_, index) => {
        const point = { x: index % width, y: Math.floor(index / width) };
        const directions = route.get(cellKey(point));
        return (
          <span className="route-cell" key={`${phase}-${cellKey(point)}`}>
            {directions?.size ? <i className="route-node" /> : null}
            {directions
              ? Array.from(directions).map((direction) => (
                  <i className={`route-branch route-${direction}`} key={direction} />
                ))
              : null}
          </span>
        );
      })}
    </div>
  );
}

function MapThumbnail({ demoCase }: { demoCase: DemoCase }) {
  const scenario = demoCase.scenario;
  const obstacles = new Set(scenario.obstacles.map(cellKey));
  const chargers = new Set(scenario.chargingStations.map(cellKey));
  return (
    <span
      className="map-thumbnail"
      style={{
        gridTemplateColumns: `repeat(${scenario.width}, 1fr)`,
        gridTemplateRows: `repeat(${scenario.height}, 1fr)`,
      }}
      aria-hidden="true"
    >
      {Array.from({ length: scenario.width * scenario.height }).map((_, index) => {
        const point = { x: index % scenario.width, y: Math.floor(index / scenario.width) };
        const key = cellKey(point);
        return (
          <i
            className={`${obstacles.has(key) ? "is-obstacle" : ""} ${chargers.has(key) ? "is-charger" : ""}`}
            key={key}
          />
        );
      })}
    </span>
  );
}

export default function Home() {
  const [bundle, setBundle] = useState<DemoBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [caseId, setCaseId] = useState("");
  const [agentId, setAgentId] = useState("");
  const [referenceId, setReferenceId] = useState("");
  const [time, setTime] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState<(typeof playbackRates)[number]>(1);
  const [showRecordedRemainder, setShowRecordedRemainder] = useState(false);

  useEffect(() => {
    fetch("./demo-data.json")
      .then((response) => {
        if (!response.ok) throw new Error(`demo gallery returned ${response.status}`);
        return response.json() as Promise<DemoBundle>;
      })
      .then((payload) => {
        if (
          payload.schemaVersion !== 2 ||
          !Number.isInteger(payload.rootSeed) ||
          !payload.cases?.length
        ) {
          throw new Error("demo gallery schema is not supported");
        }
        const requestedCase = new URLSearchParams(window.location.search).get("case");
        const initialCase =
          payload.cases.find((candidate) => candidate.caseId === requestedCase) ??
          payload.cases.find((candidate) => candidate.caseId === payload.defaultCaseId) ??
          payload.cases[0];
        const initialAgent =
          initialCase.agents.find((candidate) => candidate.id === "hybrid") ??
          initialCase.agents[0];
        setBundle(payload);
        setCaseId(initialCase.caseId);
        setAgentId(initialAgent?.id ?? "");
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "could not load demo gallery");
      });
  }, []);

  const selectedCase =
    bundle?.cases.find((candidate) => candidate.caseId === caseId) ?? bundle?.cases[0] ?? null;
  const agent =
    selectedCase?.agents.find((candidate) => candidate.id === agentId) ??
    selectedCase?.agents[0] ??
    null;
  const reference =
    selectedCase?.agents.find((candidate) => candidate.id === referenceId) ?? null;
  const scenario = selectedCase?.scenario ?? null;
  const maximumTime = scenario ? replayEndTime(scenario, agent, reference) : 1;

  useEffect(() => {
    if (!playing || maximumTime <= 1) return;
    const timer = window.setInterval(() => {
      setTime((current) => {
        if (current >= maximumTime) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, Math.round(520 / playbackRate));
    return () => window.clearInterval(timer);
  }, [playing, maximumTime, playbackRate]);

  const selectCase = (nextCaseId: string) => {
    if (!bundle) return;
    const nextCase = bundle.cases.find((candidate) => candidate.caseId === nextCaseId);
    if (!nextCase) return;
    const nextAgent =
      nextCase.agents.find((candidate) => candidate.id === agentId) ??
      nextCase.agents.find((candidate) => candidate.id === "hybrid") ??
      nextCase.agents[0];
    const nextReference = nextCase.agents.some(
      (candidate) => candidate.id === referenceId && candidate.id !== nextAgent?.id,
    )
      ? referenceId
      : "";

    setCaseId(nextCase.caseId);
    setAgentId(nextAgent?.id ?? "");
    setReferenceId(nextReference);
    setTime(1);
    setPlaying(false);
    const url = new URL(window.location.href);
    url.searchParams.set("case", nextCase.caseId);
    window.history.replaceState(null, "", url);
  };

  const selectAgent = (nextAgentId: string) => {
    if (!selectedCase || !scenario) return;
    const nextAgent = selectedCase.agents.find((candidate) => candidate.id === nextAgentId);
    if (!nextAgent) return;
    const nextReferenceId = nextAgentId === referenceId ? agentId : referenceId;
    const nextReference =
      selectedCase.agents.find((candidate) => candidate.id === nextReferenceId) ?? null;
    setReferenceId(nextReferenceId);
    setAgentId(nextAgentId);
    setTime((current) => Math.min(current, replayEndTime(scenario, nextAgent, nextReference)));
    setPlaying(false);
  };

  const selectReference = (nextReferenceId: string) => {
    if (!selectedCase || !scenario) return;
    const nextReference =
      selectedCase.agents.find((candidate) => candidate.id === nextReferenceId) ?? null;
    setReferenceId(nextReferenceId);
    setTime((current) => Math.min(current, replayEndTime(scenario, agent, nextReference)));
    setPlaying(false);
  };

  const seek = (nextTime: number) => {
    setPlaying(false);
    setTime(Math.min(maximumTime, Math.max(1, nextTime)));
  };

  if (error) {
    return (
      <main className="loading-shell error-shell">
        <p className="eyebrow">Adaptive Agent Lab</p>
        <h1>Replay gallery unavailable</h1>
        <p>{error}</p>
        <a href="https://github.com/Revincxt/adaptive-agent">Open the repository</a>
      </main>
    );
  }

  if (!bundle || !selectedCase || !scenario || !agent || !agent.trace.length) {
    return (
      <main className="loading-shell" aria-live="polite">
        <span className="loading-mark" aria-hidden="true">Adaptive Agent Lab</span>
        <p>Loading experiment gallery…</p>
      </main>
    );
  }

  const currentStep = stepAtTime(agent, time);
  const referenceStep = reference?.trace.length ? stepAtTime(reference, time) : null;
  const agentEndTime = endTime(agent);
  const referenceEndTime = endTime(reference);
  const primaryAtTerminal = time >= agentEndTime;
  const primaryPastEnd = time > agentEndTime;
  const referenceAtTerminal = Boolean(reference && time >= referenceEndTime);
  const referencePastEnd = Boolean(reference && time > referenceEndTime);
  const primaryStateTime = Math.min(time, agentEndTime);
  const robotPosition = pointFromTrace(currentStep);
  const referencePosition = referenceStep ? pointFromTrace(referenceStep) : null;
  const primaryTravelled = [
    scenario.initialRobot,
    ...agent.trace.filter((step) => step.time <= time).map(pointFromTrace),
  ];
  const primaryRemainder = [
    primaryTravelled.at(-1) ?? scenario.initialRobot,
    ...agent.trace.filter((step) => step.time > time).slice(0, 20).map(pointFromTrace),
  ];
  const referenceTravelled = reference
    ? [
        scenario.initialRobot,
        ...reference.trace.filter((step) => step.time <= time).map(pointFromTrace),
      ]
    : [];
  const deliveredOrderIds = new Set(
    agent.trace
      .filter((step) => step.time <= primaryStateTime)
      .map((step) => step.deliveredOrderId)
      .filter((orderId): orderId is string => orderId !== null),
  );
  const blocked = blockedCells(scenario.events, time);
  const obstacleSet = new Set(scenario.obstacles.map(cellKey));
  const chargerSet = new Set(scenario.chargingStations.map(cellKey));
  const batteryPercent = Math.max(
    0,
    Math.min(100, (currentStep.battery / scenario.batteryCapacity) * 100),
  );
  const completedPercent = maximumTime > 1 ? ((time - 1) / (maximumTime - 1)) * 100 : 100;
  const obstacleDensity =
    scenario.obstacles.length / Math.max(1, scenario.width * scenario.height);
  const closureCount = scenario.events.filter((event) => event.kind === "cell_blocked").length;
  const activeStyle = {
    "--agent-color": agent.color,
    "--reference-color": reference?.color ?? "#667078",
    "--agent-route-color": routeDisplayColors[agent.id] ?? agent.color,
    "--reference-route-color": reference
      ? (routeDisplayColors[reference.id] ?? reference.color)
      : "#515b63",
  } as CSSProperties;

  const orderState = (order: Order): OrderState => {
    if (deliveredOrderIds.has(order.id)) return "delivered";
    if (currentStep.carriedOrderId === order.id) return "carried";
    if (order.releaseTime > primaryStateTime) return "queued";
    if (primaryStateTime >= scenario.horizon) return "expired";
    return "ready";
  };

  const orderStates = scenario.orders.map(orderState);
  const deliveredOrderCount = orderStates.filter((state) => state === "delivered").length;
  const carriedOrderCount = orderStates.filter((state) => state === "carried").length;
  const readyOrderCount = orderStates.filter((state) => state === "ready").length;
  const queuedOrderCount = orderStates.filter((state) => state === "queued").length;
  const stateStatus = primaryAtTerminal
    ? { label: `Trace complete · t=${agentEndTime}`, tone: "complete" }
    : currentStep.violations.length
      ? { label: `${currentStep.violations.length} constraint flag(s)`, tone: "alert" }
      : batteryPercent <= 20
        ? { label: "Low battery", tone: "warning" }
        : { label: "State valid", tone: "ok" };

  return (
    <main className="app-shell" style={activeStyle}>
      <header className="app-header">
        <a className="brand" href="#workspace" aria-label="Adaptive Agent Lab home">
          <span>Adaptive Agent Lab</span>
          <strong>Replay Explorer</strong>
        </a>
        <div className="header-meta">
          <span className="recorded-status"><i /> Recorded demonstrations</span>
          <a href="#method">Method</a>
          <a href="https://github.com/Revincxt/adaptive-agent">GitHub ↗</a>
        </div>
      </header>

      <div className="workspace" id="workspace">
        <aside className="scenario-rail" aria-labelledby="scenario-library-title">
          <header className="rail-heading">
            <div>
              <h2 id="scenario-library-title">Scenario library</h2>
            </div>
            <span>{bundle.cases.length} cases</span>
          </header>

          <div className="scenario-list">
            {bundle.cases.map((candidate, index) => {
              const isSelected = candidate.caseId === selectedCase.caseId;
              const dynamics = candidate.scenario.events.filter(
                (event) => event.kind === "cell_blocked",
              ).length;
              return (
                <button
                  className={`scenario-option ${isSelected ? "is-selected" : ""}`}
                  onClick={() => selectCase(candidate.caseId)}
                  aria-pressed={isSelected}
                  key={candidate.caseId}
                >
                  <MapThumbnail demoCase={candidate} />
                  <span className="scenario-copy">
                    <small>Case {String(index + 1).padStart(2, "0")}</small>
                    <strong>{candidate.label}</strong>
                    <span>{candidate.display?.topology ?? candidate.tags[0] ?? "Warehouse layout"}</span>
                    <i>
                      {candidate.scenario.width}×{candidate.scenario.height}
                      <b>·</b>
                      {candidate.scenario.orders.length} orders
                      <b>·</b>
                      {dynamics} closures
                    </i>
                  </span>
                </button>
              );
            })}
          </div>

          <p className="rail-note">
            Four independent demonstration cases. Controller values are comparable only within the
            selected case.
          </p>
        </aside>

        <article className="experiment-view">
          <header className="experiment-heading">
            <div className="case-path">
              <span>{selectedCase.caseId}</span>
              <i aria-hidden="true">/</i>
              <span>{selectedCase.display?.difficulty ?? "Recorded case"}</span>
            </div>
            <div className="heading-row">
              <div>
                <h1>{selectedCase.label}</h1>
                <p>{selectedCase.description}</p>
              </div>
              <dl className="case-facts">
                <div><dt>Grid</dt><dd>{scenario.width} × {scenario.height}</dd></div>
                <div><dt>Obstacle density</dt><dd>{formatPercent(obstacleDensity)}</dd></div>
                <div><dt>Orders</dt><dd>{scenario.orders.length}</dd></div>
                <div><dt>Closure pairs</dt><dd>{closureCount}</dd></div>
                <div><dt>Horizon</dt><dd>{scenario.horizon}</dd></div>
              </dl>
            </div>
            <div className="scope-line" role="note">
              <strong>Scope</strong>
              <span>{bundle.verificationStatus}. Inspect behavior within a case; do not read these tapes as a benchmark ranking.</span>
            </div>
          </header>

          <section className="replay-section" aria-labelledby="replay-title">
            <div className="control-bar">
              <div className="control-title">
                <h2 id="replay-title">Trajectory and state</h2>
              </div>
              <label className="field-control">
                <span>Primary controller</span>
                <select value={agent.id} onChange={(event) => selectAgent(event.target.value)}>
                  {selectedCase.agents.map((candidate) => (
                    <option value={candidate.id} key={candidate.id}>{candidate.label}</option>
                  ))}
                </select>
              </label>
              <label className="field-control">
                <span>Compare with</span>
                <select
                  value={reference?.id ?? ""}
                  onChange={(event) => selectReference(event.target.value)}
                >
                  <option value="">None</option>
                  {selectedCase.agents
                    .filter((candidate) => candidate.id !== agent.id)
                    .map((candidate) => (
                      <option value={candidate.id} key={candidate.id}>{candidate.label}</option>
                    ))}
                </select>
              </label>
              <label className="toggle-control">
                <input
                  type="checkbox"
                  checked={showRecordedRemainder}
                  onChange={(event) => setShowRecordedRemainder(event.target.checked)}
                />
                <span>Show next 20 recorded steps</span>
              </label>
            </div>

            <div className="analysis-grid">
              <figure className="map-panel">
                <header className="panel-heading">
                  <div>
                    <span>{selectedCase.display?.topology ?? "Warehouse topology"}</span>
                    <strong>Map state at t = {time}</strong>
                  </div>
                  <div className="time-readout">
                    <span>episode</span>
                    <strong>{String(time).padStart(3, "0")}</strong>
                    <i>/ {maximumTime}</i>
                  </div>
                </header>

                <div className="map-stage">
                  <div
                    className="warehouse-map"
                    style={{ aspectRatio: `${scenario.width} / ${scenario.height}` }}
                    role="img"
                    aria-label={`${selectedCase.label}, ${scenario.width} by ${scenario.height} warehouse map at time ${time}. Primary robot at column ${robotPosition.x}, row ${robotPosition.y}. ${blocked.size} aisle closures active. ${deliveredOrderCount} orders delivered, ${carriedOrderCount} carried, ${readyOrderCount} ready, and ${queuedOrderCount} queued.`}
                  >
                    <div
                      className="map-grid"
                      style={{
                        gridTemplateColumns: `repeat(${scenario.width}, 1fr)`,
                        gridTemplateRows: `repeat(${scenario.height}, 1fr)`,
                      }}
                      aria-hidden="true"
                    >
                      {Array.from({ length: scenario.width * scenario.height }).map((_, index) => {
                        const point = {
                          x: index % scenario.width,
                          y: Math.floor(index / scenario.width),
                        };
                        const key = cellKey(point);
                        const pickups = scenario.orders.filter(
                          (order) => cellKey(order.pickup) === key,
                        );
                        const dropoffs = scenario.orders.filter(
                          (order) => cellKey(order.dropoff) === key,
                        );
                        const isPrimaryRobot = cellKey(robotPosition) === key;
                        const isReferenceRobot =
                          referencePosition && cellKey(referencePosition) === key;

                        return (
                          <span
                            className={`map-cell ${obstacleSet.has(key) ? "obstacle" : ""} ${blocked.has(key) ? "blocked" : ""}`}
                            key={key}
                          >
                            {chargerSet.has(key) ? <span className="charger-marker">C</span> : null}
                            {pickups.map((order) => (
                              <span
                                className={`order-marker pickup-marker is-${orderState(order)}`}
                                key={`pickup-${order.id}`}
                              >
                                P{scenario.orders.indexOf(order) + 1}
                              </span>
                            ))}
                            {dropoffs.map((order) => (
                              <span
                                className={`order-marker dropoff-marker is-${orderState(order)}`}
                                key={`dropoff-${order.id}`}
                              >
                                D{scenario.orders.indexOf(order) + 1}
                              </span>
                            ))}
                            {blocked.has(key) ? <span className="closure-marker">×</span> : null}
                            {isReferenceRobot ? (
                              <span
                                className={`robot-marker reference-robot ${isPrimaryRobot ? "is-overlap" : ""} ${referenceAtTerminal ? "is-trace-complete" : ""}`}
                              >
                                B
                              </span>
                            ) : null}
                            {isPrimaryRobot ? (
                              <span
                                className={`robot-marker primary-robot ${isReferenceRobot ? "is-overlap" : ""} ${primaryAtTerminal ? "is-trace-complete" : ""}`}
                              >
                                A
                              </span>
                            ) : null}
                          </span>
                        );
                      })}
                    </div>
                    {showRecordedRemainder && primaryRemainder.length > 1 ? (
                      <RouteLayer
                        points={primaryRemainder}
                        width={scenario.width}
                        height={scenario.height}
                        phase="recorded"
                      />
                    ) : null}
                    {referenceTravelled.length > 1 ? (
                      <RouteLayer
                        points={referenceTravelled}
                        width={scenario.width}
                        height={scenario.height}
                        phase="reference"
                      />
                    ) : null}
                    <RouteLayer
                      points={primaryTravelled}
                      width={scenario.width}
                      height={scenario.height}
                      phase="primary"
                    />
                  </div>
                </div>

                <div className="map-legend" aria-label="Map legend">
                  <span><i className="legend-primary" />A · {agent.label}</span>
                  {reference ? <span><i className="legend-reference" />B · {reference.label}</span> : null}
                  {showRecordedRemainder ? <span><i className="legend-recorded" />future A · next ≤20</span> : null}
                  <span><i className="legend-order legend-pickup" />P · pickup</span>
                  <span><i className="legend-order legend-dropoff" />D · drop-off</span>
                  <span><i className="legend-inactive" />faint · inactive order</span>
                  <span><i className="legend-charger" />charger</span>
                  <span><i className="legend-closure" />temporary closure</span>
                </div>

                <div className="replay-controls">
                  <div className="transport-controls" role="group" aria-label="Replay transport">
                    <button onClick={() => seek(time - 1)} disabled={time <= 1} aria-label="Previous time step">−1</button>
                    <button
                      className="play-button"
                      onClick={() => {
                        if (time >= maximumTime) setTime(1);
                        setPlaying((value) => !value);
                      }}
                      aria-label={playing ? "Pause replay" : time >= maximumTime ? "Replay from start" : "Play replay"}
                    >
                      {playing ? "Pause" : time >= maximumTime ? "Replay" : "Play"}
                    </button>
                    <button onClick={() => seek(time + 1)} disabled={time >= maximumTime} aria-label="Next time step">+1</button>
                  </div>

                  <div className="timeline-control">
                    <input
                      aria-label={`Replay time, ${time} of ${maximumTime}`}
                      aria-valuetext={`t = ${time} of ${maximumTime}`}
                      type="range"
                      min="1"
                      max={maximumTime}
                      value={time}
                      style={{ "--timeline-progress": `${completedPercent}%` } as CSSProperties}
                      onChange={(event) => seek(Number(event.target.value))}
                    />
                    <div className="timeline-events" role="group" aria-label="Scenario event shortcuts">
                      {scenario.events
                        .filter((event) => event.time <= maximumTime)
                        .map((event, index) => (
                          <button
                            key={`${event.kind}-${event.time}-${index}`}
                            className={`timeline-event event-${event.kind}`}
                            style={{
                              left: `${((event.time - 1) / Math.max(1, maximumTime - 1)) * 100}%`,
                              "--event-lane": index % 2,
                            } as CSSProperties}
                            onClick={() => seek(event.time)}
                            aria-label={`${eventLabels[event.kind]} at time ${event.time}`}
                            title={`${eventLabels[event.kind]} · t=${event.time}`}
                          />
                        ))}
                    </div>
                    <div className="timeline-scale"><span>t = 1</span><span>t = {maximumTime}</span></div>
                  </div>

                  <div className="speed-control" role="group" aria-label="Playback speed">
                    {playbackRates.map((rate) => (
                      <button
                        className={rate === playbackRate ? "is-active" : ""}
                        onClick={() => setPlaybackRate(rate)}
                        aria-pressed={rate === playbackRate}
                        key={rate}
                      >
                        {rate}×
                      </button>
                    ))}
                  </div>
                </div>

                <figcaption>
                  Solid route: executed A trace. Dashed route: synchronized B trace. The optional
                  dotted line previews at most 20 future steps from A&apos;s recorded tape.
                </figcaption>
              </figure>

              <aside className="inspector-panel" aria-labelledby="inspector-title">
                <header className="inspector-heading">
                  <div>
                    <span className="agent-dot" />
                    <div><small>{agent.family}</small><h3 id="inspector-title">{agent.label}</h3></div>
                  </div>
                  <span className={`state-badge is-${stateStatus.tone}`}>{stateStatus.label}</span>
                </header>

                <p className="agent-description">{agent.description}</p>

                <section className="state-block">
                  <h4>{primaryAtTerminal ? `Final state · trace ended at t = ${agentEndTime}` : `State at t = ${time}`}</h4>
                  <dl className="state-table">
                    <div><dt>Position</dt><dd>({robotPosition.x}, {robotPosition.y})</dd></div>
                    <div><dt>Applied action</dt><dd>{primaryPastEnd ? "—" : (actionLabels[currentStep.action] ?? currentStep.action)}</dd></div>
                    <div><dt>Payload</dt><dd>{currentStep.carriedOrderId ?? "None"}</dd></div>
                    <div><dt>Step reward</dt><dd>{primaryPastEnd ? "—" : currentStep.reward.toFixed(2)}</dd></div>
                    <div><dt>{primaryAtTerminal ? "Final return" : "Cumulative return"}</dt><dd>{currentStep.cumulativeReward.toFixed(2)}</dd></div>
                    <div><dt>Observed events</dt><dd>{primaryPastEnd ? "—" : currentStep.eventCount}</dd></div>
                  </dl>
                </section>

                <section className="battery-block">
                  <div><h4>Battery</h4><span>{currentStep.battery} / {scenario.batteryCapacity}</span></div>
                  <div className="battery-track" role="meter" aria-label="Battery level" aria-valuemin={0} aria-valuemax={scenario.batteryCapacity} aria-valuenow={currentStep.battery}>
                    <i style={{ width: `${batteryPercent}%` }} />
                  </div>
                </section>

                <section className="order-block">
                  <h4>Order lifecycle</h4>
                  <div className="order-counts">
                    <div><strong>{deliveredOrderCount}</strong><span>delivered</span></div>
                    <div><strong>{carriedOrderCount}</strong><span>carried</span></div>
                    <div><strong>{readyOrderCount}</strong><span>ready</span></div>
                    <div><strong>{queuedOrderCount}</strong><span>queued</span></div>
                  </div>
                </section>

                {reference && referenceStep && referencePosition ? (
                  <section className="comparison-state">
                    <header>
                      <span className="reference-dot" />
                      <div>
                        <small>{referenceAtTerminal ? `Trace complete · t=${referenceEndTime}` : "Comparison B"}</small>
                        <strong>{reference.label}</strong>
                      </div>
                    </header>
                    <dl>
                      <div><dt>Position</dt><dd>({referencePosition.x}, {referencePosition.y})</dd></div>
                      <div><dt>Action</dt><dd>{referencePastEnd ? "—" : (actionLabels[referenceStep.action] ?? referenceStep.action)}</dd></div>
                      <div><dt>Battery</dt><dd>{referenceStep.battery}</dd></div>
                      <div>
                        <dt>
                          Return Δ A−B
                          <small>{primaryAtTerminal ? "final" : `t=${time}`} vs {referenceAtTerminal ? "final" : `t=${time}`}</small>
                        </dt>
                        <dd>{formatSigned(currentStep.cumulativeReward - referenceStep.cumulativeReward)}</dd>
                      </div>
                    </dl>
                  </section>
                ) : (
                  <p className="comparison-empty">Choose a comparison controller to inspect two traces at the same simulator time.</p>
                )}
              </aside>
            </div>
          </section>

          <section className="results-section" aria-labelledby="results-title">
            <header className="section-heading">
              <div>
                <h2 id="results-title">Controller outcomes</h2>
              </div>
              <p>Same map, order schedule, and event tape. Values remain descriptive.</p>
            </header>

            <div className="results-table-wrap">
              <table className="results-table">
                <caption>Controller outcomes for {selectedCase.label}</caption>
                <thead>
                  <tr>
                    <th>Controller</th>
                    <th>On time</th>
                    <th>Completion</th>
                    <th>Return</th>
                    <th>Steps</th>
                    <th>Violations</th>
                    <th>Decision timing</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedCase.agents.map((candidate) => (
                    <tr
                      className={`${candidate.id === agent.id ? "is-primary" : ""} ${candidate.id === reference?.id ? "is-reference" : ""}`}
                      key={candidate.id}
                    >
                      <th scope="row">
                        <i style={{ backgroundColor: candidate.color }} />
                        <span><strong>{candidate.label}</strong><small>{candidate.family}</small></span>
                        {candidate.id === agent.id ? <b>A</b> : candidate.id === reference?.id ? <b>B</b> : null}
                      </th>
                      <td>
                        <span className="metric-value">{formatPercent(candidate.metrics.weightedOnTimeCompletionRate)}</span>
                        <span className="metric-track"><i style={{ width: formatPercent(candidate.metrics.weightedOnTimeCompletionRate), backgroundColor: candidate.color }} /></span>
                      </td>
                      <td>{formatPercent(candidate.metrics.weightedCompletionRate)}</td>
                      <td>{formatNumber(candidate.metrics.totalReward)}</td>
                      <td>{candidate.metrics.steps}</td>
                      <td>{candidate.metrics.constraintViolations}</td>
                      <td>{candidate.metrics.decisionTimeMs === null ? "Not measured" : `${formatNumber(candidate.metrics.decisionTimeMs)} ms`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="supporting-section" id="method">
            <details>
              <summary>
                <span><strong>Environment events</strong><small>{scenario.events.length} recorded events</small></span>
                <i>+</i>
              </summary>
              <div className="details-table-wrap">
                <table>
                  <thead><tr><th>Time</th><th>Event</th><th>Object / coordinate</th><th>Status</th></tr></thead>
                  <tbody>
                    {scenario.events.map((event, index) => {
                      const detail =
                        event.orderId ??
                        (event.position ? `(${event.position.x}, ${event.position.y})` : "—");
                      return (
                        <tr key={`${event.kind}-${event.time}-${index}`}>
                          <td>t = {event.time}</td>
                          <td>{eventLabels[event.kind]}</td>
                          <td>{detail}</td>
                          <td>{event.time <= time ? "Observed" : "Pending"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </details>

            <details>
              <summary>
                <span><strong>Method and provenance</strong><small>Scope, generation, and fingerprint</small></span>
                <i>+</i>
              </summary>
              <div className="provenance-grid">
                <p>
                  Each controller is trained and replayed independently for this case. The browser
                  renders committed simulator transitions; it does not run a live policy. Decision
                  timing is intentionally not measured in these portable demonstration tapes.
                </p>
                <dl>
                  <div><dt>Case ID</dt><dd>{selectedCase.caseId}</dd></div>
                  <div><dt>Map ID</dt><dd>{selectedCase.mapId}</dd></div>
                  <div><dt>Root seed</dt><dd>{bundle.rootSeed}</dd></div>
                  <div><dt>Generated</dt><dd>{new Date(bundle.generatedAt).toLocaleDateString("en-GB")}</dd></div>
                  <div><dt>Fingerprint</dt><dd><code>{selectedCase.scenarioFingerprint}</code></dd></div>
                </dl>
              </div>
            </details>
          </section>

          <footer className="app-footer">
            <span>Adaptive Agent Lab · schema v{bundle.schemaVersion}</span>
            <a href="https://github.com/Revincxt/adaptive-agent">Source, protocol, and reproducibility notes ↗</a>
          </footer>
        </article>
      </div>
    </main>
  );
}

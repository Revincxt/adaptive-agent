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
  cumulativeReward: number;
  carriedOrderId: string | null;
  deliveredOrderId: string | null;
  violations: string[];
};
type Metrics = {
  weightedOnTimeCompletionRate: number;
  totalReward: number;
  completedOrders: number;
  totalOrders: number;
  constraintViolations: number;
};
type AgentResult = {
  id: string;
  label: string;
  family: string;
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
  label: string;
  display?: {
    topology?: string;
    difficulty?: string;
    [key: string]: unknown;
  };
  scenario: ScenarioData;
  agents: AgentResult[];
};
type DemoBundle = {
  schemaVersion: number;
  rootSeed: number;
  defaultCaseId: string;
  cases: DemoCase[];
};
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
  planning: "#718095",
  replanning: "#407dc7",
  "q-learning": "#b58527",
  "dyna-q": "#936cc6",
  dqn: "#cb7651",
  hybrid: "#168f79",
};
const cellKey = (point: Point) => `${point.x}:${point.y}`;

const iconPaths = {
  grid: "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z",
  layers: "m12 3 10 6-10 6L2 9l10-6Zm-10 12 10 6 10-6M2 12l10 6 10-6",
  box: "m12 3 9 5v9l-9 5-9-5V8l9-5Zm0 10v9M3 8l9 5 9-5M7.5 5.5l9 5",
  clock: "M12 8v5l3 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z",
  route:
    "M5 17V7a3 3 0 0 1 3-3h8a3 3 0 0 1 0 6H9a3 3 0 0 0 0 6h10m-3-3 3 3-3 3M7 19a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z",
  play: "m9 5 11 7-11 7V5Z",
  pause: "M8 5v14M16 5v14",
  back: "m15 5-9 7 9 7M4 5v14",
  next: "m9 5 9 7-9 7M20 5v14",
  bolt: "m13 2-9 12h7l-1 8 10-13h-7l1-7Z",
  check: "m5 12 4 4L19 6",
  arrow: "M7 17 17 7M7 7h10v10",
  chart: "M4 3v17h17M8 15v-4M13 15V6M18 15V9",
  robot:
    "M8 7h8a3 3 0 0 1 3 3v7a3 3 0 0 1-3 3H8a3 3 0 0 1-3-3v-7a3 3 0 0 1 3-3ZM12 3v4M2 11v5M22 11v5M9 12v1M15 12v1M9 17h6",
  code: "m8 6-6 6 6 6m8-12 6 6-6 6M14 3l-4 18",
};

function Icon({
  name,
  className = "",
}: {
  name: keyof typeof iconPaths;
  className?: string;
}) {
  return (
    <svg
      className={`icon ${className}`}
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={iconPaths[name]} />
    </svg>
  );
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en", { maximumFractionDigits: 1 }).format(
    value,
  );
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
  const latestEvent = Math.max(
    1,
    ...scenario.events.map((event) => event.time),
  );
  return Math.min(
    scenario.horizon,
    Math.max(endTime(primary), endTime(reference), latestEvent),
  );
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
  return (
    <svg
      className={`route-layer route-${phase}`}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <polyline
        points={points
          .map((point) => `${point.x + 0.5},${point.y + 0.5}`)
          .join(" ")}
        fill="none"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
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
      {Array.from({ length: scenario.width * scenario.height }).map(
        (_, index) => {
          const point = {
            x: index % scenario.width,
            y: Math.floor(index / scenario.width),
          };
          const key = cellKey(point);
          return (
            <i
              className={`${obstacles.has(key) ? "is-obstacle" : ""} ${chargers.has(key) ? "is-charger" : ""}`}
              key={key}
            />
          );
        },
      )}
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
  const [playbackRate, setPlaybackRate] =
    useState<(typeof playbackRates)[number]>(1);
  const [showRecordedRemainder, setShowRecordedRemainder] = useState(false);

  useEffect(() => {
    fetch("./demo-data.json")
      .then((response) => {
        if (!response.ok)
          throw new Error(`demo gallery returned ${response.status}`);
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
        const requestedCase = new URLSearchParams(window.location.search).get(
          "case",
        );
        const initialCase =
          payload.cases.find(
            (candidate) => candidate.caseId === requestedCase,
          ) ??
          payload.cases.find(
            (candidate) => candidate.caseId === payload.defaultCaseId,
          ) ??
          payload.cases[0];
        const initialAgent =
          initialCase.agents.find((candidate) => candidate.id === "hybrid") ??
          initialCase.agents[0];
        setBundle(payload);
        setCaseId(initialCase.caseId);
        setAgentId(initialAgent?.id ?? "");
      })
      .catch((reason: unknown) => {
        setError(
          reason instanceof Error
            ? reason.message
            : "could not load demo gallery",
        );
      });
  }, []);

  const selectedCase =
    bundle?.cases.find((candidate) => candidate.caseId === caseId) ??
    bundle?.cases[0] ??
    null;
  const agent =
    selectedCase?.agents.find((candidate) => candidate.id === agentId) ??
    selectedCase?.agents[0] ??
    null;
  const reference =
    selectedCase?.agents.find((candidate) => candidate.id === referenceId) ??
    null;
  const scenario = selectedCase?.scenario ?? null;
  const maximumTime = scenario ? replayEndTime(scenario, agent, reference) : 1;

  useEffect(() => {
    if (!playing || maximumTime <= 1) return;
    const timer = window.setInterval(
      () => {
        setTime((current) => {
          if (current >= maximumTime) {
            setPlaying(false);
            return current;
          }
          return current + 1;
        });
      },
      Math.round(520 / playbackRate),
    );
    return () => window.clearInterval(timer);
  }, [playing, maximumTime, playbackRate]);

  const selectCase = (nextCaseId: string) => {
    if (!bundle) return;
    const nextCase = bundle.cases.find(
      (candidate) => candidate.caseId === nextCaseId,
    );
    if (!nextCase) return;
    const nextAgent =
      nextCase.agents.find((candidate) => candidate.id === agentId) ??
      nextCase.agents.find((candidate) => candidate.id === "hybrid") ??
      nextCase.agents[0];
    const nextReference = nextCase.agents.some(
      (candidate) =>
        candidate.id === referenceId && candidate.id !== nextAgent?.id,
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
    const nextAgent = selectedCase.agents.find(
      (candidate) => candidate.id === nextAgentId,
    );
    if (!nextAgent) return;
    const nextReferenceId = nextAgentId === referenceId ? agentId : referenceId;
    const nextReference =
      selectedCase.agents.find(
        (candidate) => candidate.id === nextReferenceId,
      ) ?? null;
    setReferenceId(nextReferenceId);
    setAgentId(nextAgentId);
    setTime((current) =>
      Math.min(current, replayEndTime(scenario, nextAgent, nextReference)),
    );
    setPlaying(false);
  };

  const selectReference = (nextReferenceId: string) => {
    if (!selectedCase || !scenario) return;
    const nextReference =
      selectedCase.agents.find(
        (candidate) => candidate.id === nextReferenceId,
      ) ?? null;
    setReferenceId(nextReferenceId);
    setTime((current) =>
      Math.min(current, replayEndTime(scenario, agent, nextReference)),
    );
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
        <a href="https://github.com/Revincxt/adaptive-agent">
          Open the repository
        </a>
      </main>
    );
  }

  if (!bundle || !selectedCase || !scenario || !agent || !agent.trace.length) {
    return (
      <main className="loading-shell" aria-live="polite">
        <span className="loading-mark" aria-hidden="true">
          Adaptive Agent Lab
        </span>
        <p>Loading experiment gallery…</p>
      </main>
    );
  }

  const currentStep = stepAtTime(agent, time);
  const referenceStep = reference?.trace.length
    ? stepAtTime(reference, time)
    : null;
  const agentEndTime = endTime(agent);
  const referenceEndTime = endTime(reference);
  const primaryAtTerminal = time >= agentEndTime;
  const primaryPastEnd = time > agentEndTime;
  const referenceAtTerminal = Boolean(reference && time >= referenceEndTime);
  const referencePastEnd = Boolean(reference && time > referenceEndTime);
  const primaryStateTime = Math.min(time, agentEndTime);
  const robotPosition = pointFromTrace(currentStep);
  const referencePosition = referenceStep
    ? pointFromTrace(referenceStep)
    : null;
  const primaryTravelled = [
    scenario.initialRobot,
    ...agent.trace.filter((step) => step.time <= time).map(pointFromTrace),
  ];
  const primaryRemainder = [
    primaryTravelled.at(-1) ?? scenario.initialRobot,
    ...agent.trace
      .filter((step) => step.time > time)
      .slice(0, 20)
      .map(pointFromTrace),
  ];
  const referenceTravelled = reference
    ? [
        scenario.initialRobot,
        ...reference.trace
          .filter((step) => step.time <= time)
          .map(pointFromTrace),
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
  const completedPercent =
    maximumTime > 1 ? ((time - 1) / (maximumTime - 1)) * 100 : 100;
  const closureCount = scenario.events.filter(
    (event) => event.kind === "cell_blocked",
  ).length;
  const activeStyle = {
    "--agent-color": routeDisplayColors[agent.id] ?? agent.color,
    "--reference-color": reference
      ? (routeDisplayColors[reference.id] ?? reference.color)
      : "#718095",
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
  const deliveredOrderCount = orderStates.filter(
    (state) => state === "delivered",
  ).length;
  const carriedOrderCount = orderStates.filter(
    (state) => state === "carried",
  ).length;
  const readyOrderCount = orderStates.filter(
    (state) => state === "ready",
  ).length;
  const queuedOrderCount = orderStates.filter(
    (state) => state === "queued",
  ).length;
  const stateStatus = primaryAtTerminal
    ? { label: "Trace complete", tone: "complete" }
    : currentStep.violations.length
      ? {
          label: `${currentStep.violations.length} constraint flag(s)`,
          tone: "alert",
        }
      : batteryPercent <= 20
        ? { label: "Low battery", tone: "warning" }
        : { label: "State valid", tone: "ok" };
  const caseNumber = String(bundle.cases.indexOf(selectedCase) + 1).padStart(
    2,
    "0",
  );

  return (
    <main className="app-shell" style={activeStyle}>
      <header className="app-header">
        <a
          className="brand"
          href="#workspace"
          aria-label="Adaptive Agent Lab home"
        >
          <span className="brand-mark">
            <Icon name="layers" />
          </span>
          <span className="brand-name">
            Adaptive<span>Agent Lab</span>
          </span>
        </a>
        <div className="header-meta">
          <span className="header-context">
            Workspace <span>/</span> Replay explorer
          </span>
          <div className="header-actions">
            <span className="recording-label">
              <Icon name="clock" />
              Recorded demo
            </span>
            <a href="https://github.com/Revincxt/adaptive-agent">
              <Icon name="code" />
              <span>GitHub</span>
              <Icon name="arrow" />
            </a>
          </div>
        </div>
      </header>

      <div className="workspace" id="workspace">
        <aside
          className="scenario-rail"
          aria-labelledby="scenario-library-title"
        >
          <header className="rail-heading">
            <div>
              <h2 id="scenario-library-title">Scenario library</h2>
            </div>
            <span>{String(bundle.cases.length).padStart(2, "0")}</span>
          </header>

          <div className="scenario-list">
            {bundle.cases.map((candidate, index) => {
              const isSelected = candidate.caseId === selectedCase.caseId;
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
                    <i>
                      {candidate.scenario.width}×{candidate.scenario.height}
                      <b>·</b>
                      {candidate.scenario.orders.length} orders
                    </i>
                  </span>
                  {isSelected ? (
                    <span className="case-selected">
                      <Icon name="check" />
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
        </aside>

        <article className="experiment-view">
          <header className="experiment-heading">
            <div className="heading-row">
              <div>
                <p className="eyebrow">
                  <span className="eyebrow-line" />
                  Scenario {caseNumber}
                </p>
                <h1>{selectedCase.label}</h1>
                <span className="difficulty-badge">
                  {selectedCase.display?.difficulty ?? "Recorded case"}
                </span>
              </div>
              <dl className="case-facts">
                <div>
                  <dt>
                    <Icon name="grid" />
                    Grid
                  </dt>
                  <dd>
                    {scenario.width} <span>×</span> {scenario.height}
                  </dd>
                </div>
                <div>
                  <dt>
                    <Icon name="box" />
                    Orders
                  </dt>
                  <dd>{String(scenario.orders.length).padStart(2, "0")}</dd>
                </div>
                <div>
                  <dt>
                    <Icon name="route" />
                    Closures
                  </dt>
                  <dd>{String(closureCount).padStart(2, "0")}</dd>
                </div>
                <div>
                  <dt>
                    <Icon name="clock" />
                    Horizon
                  </dt>
                  <dd>
                    {scenario.horizon}
                    <small>steps</small>
                  </dd>
                </div>
              </dl>
            </div>
          </header>

          <section className="replay-section" aria-labelledby="replay-title">
            <div className="control-bar">
              <div className="control-title">
                <span className="section-icon">
                  <Icon name="route" />
                </span>
                <div>
                  <h2 id="replay-title">Simulation replay</h2>
                  <span
                    className={`playback-status ${playing ? "is-playing" : ""}`}
                  >
                    <i />
                    {playing
                      ? "Playing"
                      : primaryAtTerminal
                        ? "Complete"
                        : "Paused"}
                  </span>
                </div>
              </div>
              <label className="field-control">
                <span>Primary controller</span>
                <select
                  value={agent.id}
                  onChange={(event) => selectAgent(event.target.value)}
                >
                  {selectedCase.agents.map((candidate) => (
                    <option value={candidate.id} key={candidate.id}>
                      {candidate.label}
                    </option>
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
                      <option value={candidate.id} key={candidate.id}>
                        {candidate.label}
                      </option>
                    ))}
                </select>
              </label>
              <label className="toggle-control">
                <input
                  type="checkbox"
                  checked={showRecordedRemainder}
                  onChange={(event) =>
                    setShowRecordedRemainder(event.target.checked)
                  }
                />
                <span>Future path</span>
              </label>
            </div>

            <div className="analysis-grid">
              <figure className="map-panel">
                <header className="panel-heading">
                  <div className="map-title">
                    <Icon name="grid" />
                    <strong>
                      {selectedCase.display?.topology ?? "Warehouse floor"}
                    </strong>
                  </div>
                  <div className="time-readout">
                    <span>step</span>
                    <strong>{String(time).padStart(3, "0")}</strong>
                    <i>/ {maximumTime}</i>
                  </div>
                </header>

                <div className="map-stage">
                  <div
                    className="warehouse-map"
                    style={{
                      aspectRatio: `${scenario.width} / ${scenario.height}`,
                    }}
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
                      {Array.from({
                        length: scenario.width * scenario.height,
                      }).map((_, index) => {
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
                          referencePosition &&
                          cellKey(referencePosition) === key;

                        return (
                          <span
                            className={`map-cell ${obstacleSet.has(key) ? "obstacle" : ""} ${blocked.has(key) ? "blocked" : ""}`}
                            key={key}
                          >
                            {chargerSet.has(key) ? (
                              <span className="charger-marker">
                                <Icon name="bolt" />
                              </span>
                            ) : null}
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
                            {blocked.has(key) ? (
                              <span className="closure-marker">×</span>
                            ) : null}
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
                  <span>
                    <i className="legend-primary" />A · {agent.label}
                  </span>
                  {reference ? (
                    <span>
                      <i className="legend-reference" />B · {reference.label}
                    </span>
                  ) : null}
                  {showRecordedRemainder ? (
                    <span>
                      <i className="legend-recorded" />
                      future A · next ≤20
                    </span>
                  ) : null}
                  <span>
                    <i className="legend-order legend-pickup" />P · pickup
                  </span>
                  <span>
                    <i className="legend-order legend-dropoff" />D · drop-off
                  </span>
                  <span>
                    <i className="legend-charger" />
                    charger
                  </span>
                  <span>
                    <i className="legend-closure" />
                    temporary closure
                  </span>
                </div>

                <div className="replay-controls">
                  <div
                    className="transport-controls"
                    role="group"
                    aria-label="Replay transport"
                  >
                    <button
                      onClick={() => seek(time - 1)}
                      disabled={time <= 1}
                      aria-label="Previous time step"
                    >
                      <Icon name="back" />
                    </button>
                    <button
                      className="play-button"
                      onClick={() => {
                        if (time >= maximumTime) setTime(1);
                        setPlaying((value) => !value);
                      }}
                      aria-label={
                        playing
                          ? "Pause replay"
                          : time >= maximumTime
                            ? "Replay from start"
                            : "Play replay"
                      }
                    >
                      <Icon name={playing ? "pause" : "play"} />
                      {playing
                        ? "Pause"
                        : time >= maximumTime
                          ? "Replay"
                          : "Play"}
                    </button>
                    <button
                      onClick={() => seek(time + 1)}
                      disabled={time >= maximumTime}
                      aria-label="Next time step"
                    >
                      <Icon name="next" />
                    </button>
                  </div>

                  <div className="timeline-control">
                    <input
                      aria-label={`Replay time, ${time} of ${maximumTime}`}
                      aria-valuetext={`t = ${time} of ${maximumTime}`}
                      type="range"
                      min="1"
                      max={maximumTime}
                      value={time}
                      style={
                        {
                          "--timeline-progress": `${completedPercent}%`,
                        } as CSSProperties
                      }
                      onChange={(event) => seek(Number(event.target.value))}
                    />
                    <div
                      className="timeline-events"
                      role="group"
                      aria-label="Scenario event shortcuts"
                    >
                      {scenario.events
                        .filter((event) => event.time <= maximumTime)
                        .map((event, index) => (
                          <button
                            key={`${event.kind}-${event.time}-${index}`}
                            className={`timeline-event event-${event.kind}`}
                            style={
                              {
                                left: `${((event.time - 1) / Math.max(1, maximumTime - 1)) * 100}%`,
                                "--event-lane": index % 2,
                              } as CSSProperties
                            }
                            onClick={() => seek(event.time)}
                            aria-label={`${eventLabels[event.kind]} at time ${event.time}`}
                            title={`${eventLabels[event.kind]} · t=${event.time}`}
                          />
                        ))}
                    </div>
                    <div className="timeline-scale">
                      <span>t = 1</span>
                      <span>t = {maximumTime}</span>
                    </div>
                  </div>

                  <div
                    className="speed-control"
                    role="group"
                    aria-label="Playback speed"
                  >
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
              </figure>

              <aside
                className="inspector-panel"
                aria-labelledby="inspector-title"
              >
                <header className="inspector-heading">
                  <div>
                    <span className="agent-avatar">
                      <Icon name="robot" />
                    </span>
                    <div>
                      <small>Primary controller</small>
                      <h3 id="inspector-title">{agent.label}</h3>
                    </div>
                  </div>
                  <span className={`state-badge is-${stateStatus.tone}`}>
                    <i />
                    {stateStatus.label}
                  </span>
                </header>

                <section className="state-block">
                  <h4>
                    {primaryAtTerminal
                      ? `Final state · trace ended at t = ${agentEndTime}`
                      : `State at t = ${time}`}
                  </h4>
                  <dl className="state-table">
                    <div>
                      <dt>Position</dt>
                      <dd>
                        ({robotPosition.x}, {robotPosition.y})
                      </dd>
                    </div>
                    <div>
                      <dt>Applied action</dt>
                      <dd>
                        {primaryPastEnd
                          ? "—"
                          : (actionLabels[currentStep.action] ??
                            currentStep.action)}
                      </dd>
                    </div>
                    <div>
                      <dt>Payload</dt>
                      <dd>{currentStep.carriedOrderId ?? "None"}</dd>
                    </div>
                    <div>
                      <dt>
                        {primaryAtTerminal
                          ? "Final return"
                          : "Cumulative return"}
                      </dt>
                      <dd>{currentStep.cumulativeReward.toFixed(2)}</dd>
                    </div>
                  </dl>
                </section>

                <section className="battery-block">
                  <div>
                    <h4>
                      <Icon name="bolt" />
                      Battery
                    </h4>
                    <span>
                      {Math.round(batteryPercent)}
                      <small>%</small>
                    </span>
                  </div>
                  <div
                    className={`battery-track ${batteryPercent <= 20 ? "is-low" : ""}`}
                    role="meter"
                    aria-label="Battery level"
                    aria-valuemin={0}
                    aria-valuemax={scenario.batteryCapacity}
                    aria-valuenow={currentStep.battery}
                  >
                    <i style={{ width: `${batteryPercent}%` }} />
                  </div>
                  <p className="battery-capacity">
                    {currentStep.battery} / {scenario.batteryCapacity} units
                  </p>
                </section>

                <section className="order-block">
                  <h4>Order lifecycle</h4>
                  <div className="order-counts">
                    <div className="delivered-count">
                      <strong>{deliveredOrderCount}</strong>
                      <span>delivered</span>
                    </div>
                    <div>
                      <strong>{carriedOrderCount}</strong>
                      <span>carried</span>
                    </div>
                    <div>
                      <strong>{readyOrderCount}</strong>
                      <span>ready</span>
                    </div>
                    <div>
                      <strong>{queuedOrderCount}</strong>
                      <span>queued</span>
                    </div>
                  </div>
                  <ol className="order-list">
                    {scenario.orders.map((order, index) => {
                      const state = orderStates[index];
                      return (
                        <li
                          className={`order-row is-${state}`}
                          aria-label={`Order ${index + 1}: ${state}`}
                          key={order.id}
                        >
                          <span className="order-number">
                            {String(index + 1).padStart(2, "0")}
                          </span>
                          <span className="order-status">
                            {state === "delivered" ? (
                              <Icon name="check" />
                            ) : null}
                            {state}
                          </span>
                        </li>
                      );
                    })}
                  </ol>
                </section>

                {reference && referenceStep && referencePosition ? (
                  <section className="comparison-state">
                    <header>
                      <span className="reference-dot" />
                      <div>
                        <small>
                          {referenceAtTerminal
                            ? `Trace complete · t=${referenceEndTime}`
                            : "Comparison B"}
                        </small>
                        <strong>{reference.label}</strong>
                      </div>
                    </header>
                    <dl>
                      <div>
                        <dt>Position</dt>
                        <dd>
                          ({referencePosition.x}, {referencePosition.y})
                        </dd>
                      </div>
                      <div>
                        <dt>Action</dt>
                        <dd>
                          {referencePastEnd
                            ? "—"
                            : (actionLabels[referenceStep.action] ??
                              referenceStep.action)}
                        </dd>
                      </div>
                      <div>
                        <dt>Battery</dt>
                        <dd>{referenceStep.battery}</dd>
                      </div>
                      <div>
                        <dt>
                          Return Δ A−B
                          <small>
                            {primaryAtTerminal ? "final" : `t=${time}`} vs{" "}
                            {referenceAtTerminal ? "final" : `t=${time}`}
                          </small>
                        </dt>
                        <dd>
                          {formatSigned(
                            currentStep.cumulativeReward -
                              referenceStep.cumulativeReward,
                          )}
                        </dd>
                      </div>
                    </dl>
                  </section>
                ) : null}
              </aside>
            </div>
          </section>

          <section className="results-section" aria-labelledby="results-title">
            <header className="section-heading">
              <div>
                <span className="section-icon">
                  <Icon name="chart" />
                </span>
                <h2 id="results-title">Controller outcomes</h2>
              </div>
              <span className="section-subtitle">
                Final episode results <span>·</span>{" "}
                {selectedCase.agents.length} controllers
              </span>
            </header>

            <div className="results-table-wrap">
              <table className="results-table">
                <caption>Controller outcomes for {selectedCase.label}</caption>
                <thead>
                  <tr>
                    <th>Controller</th>
                    <th>On time</th>
                    <th>Delivered</th>
                    <th>Return</th>
                    <th>Violations</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedCase.agents.map((candidate) => (
                    <tr
                      className={`${candidate.id === agent.id ? "is-primary" : ""} ${candidate.id === reference?.id ? "is-reference" : ""}`}
                      key={candidate.id}
                    >
                      <th scope="row">
                        <button
                          className="controller-option"
                          onClick={() => selectAgent(candidate.id)}
                          aria-label={`Replay ${candidate.label}`}
                          aria-pressed={candidate.id === agent.id}
                        >
                          <i
                            style={{
                              backgroundColor:
                                routeDisplayColors[candidate.id] ??
                                candidate.color,
                            }}
                          />
                          <span>
                            <strong>{candidate.label}</strong>
                            <small>{candidate.family}</small>
                          </span>
                          {candidate.id === agent.id ? (
                            <b>A</b>
                          ) : candidate.id === reference?.id ? (
                            <b>B</b>
                          ) : null}
                        </button>
                      </th>
                      <td>
                        <span className="metric-value">
                          {formatPercent(
                            candidate.metrics.weightedOnTimeCompletionRate,
                          )}
                        </span>
                        <span className="metric-track">
                          <i
                            style={{
                              width: formatPercent(
                                candidate.metrics.weightedOnTimeCompletionRate,
                              ),
                              backgroundColor:
                                routeDisplayColors[candidate.id] ??
                                candidate.color,
                            }}
                          />
                        </span>
                      </td>
                      <td>
                        {candidate.metrics.completedOrders}/
                        {candidate.metrics.totalOrders}
                      </td>
                      <td>{formatNumber(candidate.metrics.totalReward)}</td>
                      <td>
                        <span
                          className={`violation-count ${candidate.metrics.constraintViolations === 0 ? "is-clear" : ""}`}
                        >
                          {candidate.metrics.constraintViolations === 0 ? (
                            <Icon name="check" />
                          ) : null}
                          {candidate.metrics.constraintViolations}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </article>
      </div>
    </main>
  );
}

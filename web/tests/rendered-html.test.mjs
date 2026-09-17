import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the production experiment shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(
    html,
    /<title>Adaptive Agent Lab — Multi-map Replay Explorer<\/title>/i,
  );
  assert.match(html, /Loading experiment gallery/);
  assert.match(html, /aria-live="polite"/);
  assert.doesNotMatch(html, /Your site is taking shape|react-loading-skeleton/);
});

test("ships a four-map gallery with six real controller traces per case", async () => {
  const [artifactText, page, css] = await Promise.all([
    readFile(new URL("../public/demo-data.json", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  const artifact = JSON.parse(artifactText);

  assert.equal(artifact.schemaVersion, 2);
  assert.equal(artifact.rootSeed, 42);
  assert.equal(artifact.verificationStatus, "DEMO · NON-CONFIRMATORY · PAIRED TAPE");
  assert.equal(artifact.defaultCaseId, "rack-maze");
  assert.deepEqual(
    artifact.cases.map((demoCase) => demoCase.caseId),
    ["rack-maze", "parallel-aisles", "cross-dock", "serpentine"],
  );

  const expectedAgents = ["planning", "replanning", "q-learning", "dyna-q", "dqn", "hybrid"];
  const fingerprints = new Set();
  for (const demoCase of artifact.cases) {
    const { scenario } = demoCase;
    assert.match(demoCase.scenarioFingerprint, /^sha256:[0-9a-f]{64}$/);
    fingerprints.add(demoCase.scenarioFingerprint);
    assert.equal(scenario.width, 16);
    assert.equal(scenario.height, 12);
    assert.equal(scenario.horizon, 240);
    assert.equal(scenario.orders.length, 6);
    assert.equal(scenario.events.length, 9);
    assert.deepEqual(demoCase.agents.map((agent) => agent.id), expectedAgents);

    const inBounds = (point) =>
      Number.isInteger(point.x) && Number.isInteger(point.y) &&
      point.x >= 0 && point.x < scenario.width && point.y >= 0 && point.y < scenario.height;
    const obstacleKeys = new Set(scenario.obstacles.map((point) => `${point.x}:${point.y}`));
    assert.ok(inBounds(scenario.initialRobot));
    assert.ok(!obstacleKeys.has(`${scenario.initialRobot.x}:${scenario.initialRobot.y}`));
    assert.ok(scenario.obstacles.every(inBounds));
    assert.ok(scenario.chargingStations.every(inBounds));
    assert.ok(scenario.orders.every((order) => inBounds(order.pickup) && inBounds(order.dropoff)));
    assert.ok(scenario.events.every((event) =>
      event.time >= 0 && event.time <= scenario.horizon && (!event.position || inBounds(event.position))
    ));

    for (const agent of demoCase.agents) {
      assert.ok(agent.trace.length > 0);
      assert.ok(agent.trace.every((step) => inBounds({ x: step.position[0], y: step.position[1] })));
      assert.ok(agent.trace.every((step, index) => index === 0 || step.time > agent.trace[index - 1].time));
    }
  }
  assert.equal(fingerprints.size, 4);
  assert.match(page, /fetch\("\.\/demo-data\.json"\)/);
  assert.match(page, /payload\.schemaVersion !== 2/);
  assert.match(page, /Scenario library/);
  assert.match(page, /Applied action/);
  assert.match(page, /return "expired"/);
  assert.match(page, /Compare with/);
  assert.match(page, /Future path/);
  assert.match(page, /Trace complete/);
  assert.doesNotMatch(page, /decisionTimeMs|Method and provenance|Environment events/);
  assert.match(page, /slice\(0, 20\)/);
  assert.match(page, /playbackRates/);
  assert.match(page, /new URLSearchParams\(window\.location\.search\)/);
  assert.match(page, /url\.searchParams\.set\("case"/);
  assert.match(page, /type="range"/);
  assert.match(page, /selectCase/);
  assert.match(page, /selectAgent/);
  assert.match(css, /\.analysis-grid/);
  assert.match(css, /\.warehouse-map/);
  assert.doesNotMatch(page, /Mission control|Replay online|Floor 07|Figure 1|Interpretation notes/);
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
});

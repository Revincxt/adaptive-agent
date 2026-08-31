import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("exports a self-contained GitHub Pages artifact under the repository path", async () => {
  const html = await readFile(
    new URL("../dist/pages/index.html", import.meta.url),
    "utf8",
  );
  const pagesBase = "/adaptive-agent/";

  assert.match(
    html,
    /<title>Adaptive Agent Lab — Multi-map Replay Explorer<\/title>/i,
  );
  assert.match(html, /href="\/adaptive-agent\/assets\/[^" ]+\.css"/);
  assert.match(html, /import\("\/adaptive-agent\/assets\/[^" ]+\.js"\)/);
  assert.match(
    html,
    /href="https:\/\/revincxt\.github\.io\/adaptive-agent\/favicon\.svg"/,
  );
  assert.match(
    html,
    /property="og:image" content="https:\/\/revincxt\.github\.io\/adaptive-agent\/og\.png"/,
  );
  assert.match(
    html,
    /name="twitter:image" content="https:\/\/revincxt\.github\.io\/adaptive-agent\/og\.png"/,
  );
  assert.match(
    html,
    /rel="canonical" href="https:\/\/revincxt\.github\.io\/adaptive-agent\/"/,
  );
  assert.doesNotMatch(
    html,
    /(?:href|src|content)=["']\/(?:assets\/|favicon\.svg|og\.png)/,
  );
  assert.doesNotMatch(html, /import\(["']\/assets\//);
  assert.doesNotMatch(html, /adaptive-agent\/adaptive-agent/);

  const assetUrls = new Set(
    html.match(
      /\/adaptive-agent\/(?:assets\/[^"'\\\s<]+|favicon\.svg|og\.png)/g,
    ) ?? [],
  );
  assert.ok(assetUrls.size >= 7);
  await Promise.all(
    [...assetUrls].map((url) =>
      readFile(
        new URL(`../dist/pages/${url.slice(pagesBase.length)}`, import.meta.url),
      ),
    ),
  );

  const [sourceDemo, pagesDemo, sourceOg, pagesOg] = await Promise.all([
    readFile(new URL("../public/demo-data.json", import.meta.url), "utf8"),
    readFile(new URL("../dist/pages/demo-data.json", import.meta.url), "utf8"),
    readFile(new URL("../public/og.png", import.meta.url)),
    readFile(new URL("../dist/pages/og.png", import.meta.url)),
  ]);
  const exportedDemo = JSON.parse(pagesDemo);
  assert.equal(exportedDemo.rootSeed, 42);
  assert.ok(
    exportedDemo.cases.every((demoCase) =>
      demoCase.agents.every((agent) => agent.metrics.decisionTimeMs === null)
    ),
  );
  assert.deepEqual(exportedDemo, JSON.parse(sourceDemo));
  assert.deepEqual(pagesOg, sourceOg);
});

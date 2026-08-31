import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const pagesBase = process.env.GITHUB_PAGES_BASE_PATH ?? "/adaptive-agent/";

assert.match(
  pagesBase,
  /^\/(?:[A-Za-z0-9._~-]+\/)*$/,
  "GITHUB_PAGES_BASE_PATH must be an absolute, trailing-slash URL path",
);
process.env.GITHUB_PAGES_BASE_PATH = pagesBase;

const webDirectory = fileURLToPath(new URL("../", import.meta.url));
const pnpmCommand = process.platform === "win32" ? "pnpm.cmd" : "pnpm";
await new Promise((resolve, reject) => {
  const build = spawn(pnpmCommand, ["run", "build"], {
    cwd: webDirectory,
    env: { ...process.env, GITHUB_PAGES_BASE_PATH: pagesBase },
    stdio: "inherit",
  });
  build.on("error", reject);
  build.on("exit", (code, signal) => {
    if (code === 0) {
      resolve();
      return;
    }
    reject(
      new Error(
        signal
          ? `vinext build terminated by ${signal}`
          : `vinext build exited with code ${code}`,
      ),
    );
  });
});

const clientDirectory = new URL("../dist/client/", import.meta.url);
const pagesDirectory = new URL("../dist/pages/", import.meta.url);
const manifestUrl = new URL(".vite/manifest.json", clientDirectory);
const workerUrl = new URL("../dist/server/index.js", import.meta.url);
workerUrl.searchParams.set("github-pages-export", `${process.pid}-${Date.now()}`);

const { default: worker } = await import(workerUrl.href);
const response = await worker.fetch(
  new Request("https://revincxt.github.io/", {
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

assert.equal(response.status, 200, "the production worker must render the root route");
assert.match(
  response.headers.get("content-type") ?? "",
  /^text\/html\b/i,
  "the production worker must return HTML",
);

const renderedHtml = await response.text();
// Next metadata keeps public-file URLs root-relative. Rebase those references
// without changing the shared application layout used by the worker deployment.
const publicFiles = ["favicon.svg", "og.png"];
const publicFilePattern = new RegExp(
  `(["'])/(${publicFiles.map((filename) => filename.replace(".", "\\.")).join("|")})\\1`,
  "g",
);
const html = renderedHtml.replace(
  publicFilePattern,
  (_match, quote, filename) => `${quote}${pagesBase}${filename}${quote}`,
);
const publicSiteUrl = new URL(pagesBase, "https://revincxt.github.io/");
assert.match(
  html,
  /<title>Adaptive Agent Lab — Multi-map Replay Explorer<\/title>/i,
);
assert.ok(
  html.includes(`property="og:image" content="${new URL("og.png", publicSiteUrl)}"`),
  "the exported social card must use its public GitHub Pages URL",
);
assert.ok(
  html.includes(`${pagesBase}assets/`),
  `rendered asset URLs must use ${pagesBase}`,
);
assert.doesNotMatch(
  html,
  /(?:href|src)=["']\/assets\//,
  "root-relative assets would break on a project Pages site",
);
assert.doesNotMatch(
  html,
  /import\(["']\/assets\//,
  "the browser entry must use the project Pages base path",
);
assert.doesNotMatch(
  html,
  /["']\/(?:favicon\.svg|og\.png)/,
  "public files must use the project Pages base path",
);
assert.doesNotMatch(
  html,
  /adaptive-agent\/adaptive-agent/,
  "already-based public URLs must not be rebased twice",
);

const [manifestText, demoText] = await Promise.all([
  readFile(manifestUrl, "utf8"),
  readFile(new URL("demo-data.json", clientDirectory), "utf8"),
]);
const manifest = JSON.parse(manifestText);
const browserEntry = manifest["virtual:vinext-app-browser-entry"]?.file;
const pageEntry = manifest["app/page.tsx"]?.file;
assert.equal(typeof browserEntry, "string", "the browser entry must exist in the Vite manifest");
assert.equal(typeof pageEntry, "string", "the dashboard entry must exist in the Vite manifest");

const [browserEntryText, pageEntryText] = await Promise.all([
  readFile(new URL(browserEntry, clientDirectory), "utf8"),
  readFile(new URL(pageEntry, clientDirectory), "utf8"),
]);
assert.ok(
  browserEntryText.includes(pagesBase),
  "the Vite preload runtime must preserve the GitHub Pages base path",
);
assert.ok(
  pageEntryText.includes("./demo-data.json"),
  "the dashboard must load demo data relative to the project Pages URL",
);

const demo = JSON.parse(demoText);
assert.equal(
  demo.verificationStatus,
  "DEMO · NON-CONFIRMATORY · PAIRED TAPE",
  "the Pages gallery must use committed non-confirmatory artifacts",
);
assert.ok(
  demo.schemaVersion === 2 && Array.isArray(demo.cases) && demo.cases.length === 4,
  "the Pages gallery must include all four structured cases",
);
assert.equal(demo.rootSeed, 42, "the Pages gallery must record its reproducibility seed");
assert.ok(
  demo.cases.every((demoCase) => Array.isArray(demoCase.agents) && demoCase.agents.length === 6),
  "every Pages gallery case must include all six agent traces",
);
assert.ok(
  demo.cases.every((demoCase) =>
    demoCase.agents.every((agent) => agent.metrics.decisionTimeMs === null)
  ),
  "unmeasured decision timing must remain null rather than appear as zero",
);

await rm(pagesDirectory, { recursive: true, force: true });
await mkdir(pagesDirectory, { recursive: true });
await cp(clientDirectory, pagesDirectory, { recursive: true });
await Promise.all([
  writeFile(new URL("index.html", pagesDirectory), html, "utf8"),
  writeFile(new URL(".nojekyll", pagesDirectory), "", "utf8"),
]);

console.log(`Exported GitHub Pages artifact to ${pagesDirectory.pathname}`);

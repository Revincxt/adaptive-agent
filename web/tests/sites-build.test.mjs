import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("builds a self-contained root-path Sites artifact", async () => {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("sites-root", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  const requestOrigin = "https://adaptive-agent-lab.example";
  const canonicalOrigin = "https://adaptive-agent-lab.my20000806.chatgpt.site";
  const response = await worker.fetch(
    new Request(`${requestOrigin}/`, { headers: { accept: "text/html" } }),
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

  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /href="\/assets\/[^" ]+\.css"/);
  assert.match(html, /import\("\/assets\/[^" ]+\.js"\)/);
  assert.match(html, new RegExp(`rel="canonical" href="${canonicalOrigin}/"`));
  assert.match(
    html,
    new RegExp(`property="og:image" content="${canonicalOrigin}/og\\.png"`),
  );
  assert.doesNotMatch(html, /\/adaptive-agent\/assets\//);

  const assetUrls = new Set(html.match(/\/assets\/[^"'\\\s<]+/g) ?? []);
  assert.ok(assetUrls.size >= 5);
  await Promise.all(
    [...assetUrls].map((url) =>
      readFile(new URL(`../dist/client/${url.slice(1)}`, import.meta.url)),
    ),
  );

  const [demoText, socialCard] = await Promise.all([
    readFile(new URL("../dist/client/demo-data.json", import.meta.url), "utf8"),
    readFile(new URL("../dist/client/og.png", import.meta.url)),
  ]);
  const demo = JSON.parse(demoText);
  assert.equal(demo.schemaVersion, 2);
  assert.equal(demo.rootSeed, 42);
  assert.equal(demo.cases.length, 4);
  assert.ok(socialCard.byteLength > 100_000);
});

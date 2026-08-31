# Adaptive Agent Lab replay explorer

This vinext application presents a compact research interface for inspecting
warehouse trajectories and descriptive metrics for the six Adaptive Agent Lab
controllers. It reads the versioned four-case gallery in
`public/demo-data.json`, generated from real simulator transitions by the
Python package.

The bundled data is a **non-confirmatory demonstration**. Learners receive small
convenience training budgets on the displayed scenario, so the artifact is
useful for inspecting behavior and testing the replay UI—not for claiming an
algorithm ranking or a held-out benchmark result.

## Local development

Requires Node.js `>=22.13.0` and pnpm.

```bash
pnpm install
pnpm dev
```

From the repository root, regenerate all four displayed cases with:

```bash
aal export-gallery \
  --config configs/demo-gallery.json \
  --output web/public/demo-data.json \
  --seed 42
```

The checked-in gallery contains four matched 16×12 warehouse layouts: rack
islands, parallel aisles, a cross-dock spine, and a serpentine corridor. Each
case has four orders, three charging positions, and two paired aisle
closure/reopening events. Every closure preserves a valid detour.
The artifact records the root seed and per-case fingerprints; decision timing
is deliberately marked as unmeasured in the portable replay.

## Checks

```bash
pnpm lint
pnpm test
pnpm test:pages
pnpm build
```

`pnpm test` includes a production build before checking the rendered HTML.
`pnpm test:pages` additionally exports and verifies the public
`/adaptive-agent/` artifact used by GitHub Pages.

# Design.md — Volcano Monitoring Python Packages

> Purpose: seed a Claude Code session with the architecture and packaging decisions
> for a set of Python libraries used in volcano/geohazard monitoring. Decisions are
> stated with compressed rationale so they don't get re-litigated. Genuinely open
> decisions are flagged in **§10**. Start there if you're scoping work.

---

## 1. Context

A family of Python packages for volcano monitoring:

- **Data access** — fetches raw streams (FDSN/waveforms, archived products, etc.) and
  combines streams per volcano.
- **Seismic features** — computes seismic features (RSAM, SSAM, DSAR, …) from waveforms.
- **Higher-level packages** — built on top of the above, "lego-style", so that a change
  to (e.g.) data access happens in exactly one place.

Maintained solo, best-effort (not a dedicated DevOps role). Optimise for low maintenance
overhead and for making cross-cutting changes cheaply.

---

## 2. Core principle: draw boundaries along the axis of change

Do **not** partition packages by *what kind of thing the data is* ("is RSAM a feature or a
stream?"). That taxonomy question has no stable answer and makes the boundary wobble every
time a new derived quantity appears.

Partition instead by **who has to change when X changes, and which way dependencies point.**
Boundaries drawn along the axis of change stay put. This single rule drives everything below.

---

## 3. Layered architecture

```
research / product packages       ← the "lego" built on top
        │  (imports point downward only)
        ▼
composition / "volcano view"      ← per-volcano config, stream alignment,
        │            │               stored-or-compute policy
        ▼            ▼
   access  ←siblings→  features    ← neither imports the other
        │            │
        ▼            ▼
        contracts / types          ← tiny, very stable shared kernel
```

### The two load-bearing rules

1. **Dependencies point downward only.** Lower layers never import higher ones.
2. **Computation packages take data in and return results out; they never fetch.**
   `features` operates on an in-memory waveform object (ObsPy `Stream`/`Trace`, or a
   `contracts` type) and must not import `access`. `access` and `features` are **siblings**
   that never import each other.

Why rule 2 matters: the moment `features` imports `access`, you can't unit-test a feature
without standing up a data source, and the two most-reused things are welded together.
Keeping `features` I/O-free is what actually delivers the composability ("lego") goal.

---

## 4. Package responsibilities

| Package | Owns | Must NOT |
|---|---|---|
| `contracts` | Shared types/interfaces (waveform contract, event/threshold types). Tiny, stable. | Depend on anything else in the family. |
| `access` | Pure I/O: fetch raw streams and stored/archived products. One public API; internals hidden. | Compute features. Contain per-volcano `if` branches. Be imported by `features`. |
| `features` | Algorithms over in-memory waveforms (RSAM/SSAM/DSAR computation). | Fetch anything. Import `access`. |
| `composition` | Per-volcano domain aggregate: which streams belong to which volcano, time-alignment, and the **stored-or-compute policy**. The only layer allowed to depend on both `access` and `features`. | Reach around `access` to hit FDSN directly. |
| consumers | Research/product code built on top. | Be imported by any lower layer. |

**Key move:** the "combine streams per volcano" job is neither access nor feature — it's a
domain aggregate. It currently squats in `access` and drags it upward. Move it to
`composition`. That keeps `access` genuinely pure I/O, which is what lets "change data
access in one place" actually hold.

---

## 5. Worked example — RSAM (apply the rule, don't ask the taxonomy question)

There are two RSAMs hiding under one name:

- **Computing RSAM** — an algorithm over waveforms (window length, filter band, gap
  handling). Whoever tweaks the band touches this. → lives in `features`.
- **Fetching stored RSAM** — if the observatory archives/serves RSAM as a product,
  retrieving it is opaque I/O; the access layer neither knows nor cares how it was made.
  → lives in `access`, as a stream like any other.

The decision of **"fetch the stored one vs. compute on demand"** belongs in `composition`,
above both. This same pattern absorbs SSAM, DSAR, and any future metric — you never ask
"but is this one a feature?" again.

**Collapse rule:** if in *your* stack a metric is only ever fetched (never recomputed) or
only ever computed (never stored), don't build the stored-or-compute split speculatively —
put it wholly on one side. The split only earns its keep when both paths are real.

---

## 6. Per-volcano variation = configuration, not branches

The biggest lever for "single source of truth" is **not** package boundaries — it's making
the differences between volcanoes *configuration that `composition` reads*, not
`if volcano == "ruapehu"` branches sprinkled across packages.

The lego bricks are identical; only the baseplate (config) varies per volcano.

---

## 7. Distribution vs repo vs import package (don't conflate them)

Three orthogonal concepts:

- **Repo** — unit of version control. What you `git clone`.
- **Distribution** — unit of release/install. What you `pip install`; has a `pyproject.toml`
  and a dependency list.
- **Import package** — unit of import. What you `import`.

The dependency list lands on the **distribution**, not the repo. So a monorepo does **not**
mean one giant install: it holds many small distributions, each with its own narrow deps.

- A scientist who wants only features runs `pip install aitana-features` → gets
  numpy/scipy/obspy, **not** the FDSN clients or DB drivers (those live in `aitana-access`).
- The full dependency set exists only in the dev environment of someone hacking on the whole
  repo, managed by a workspace tool + single lockfile — never dumped on users.

### Decision: many distributions, one repo (for now)

- **Distribution count** is chosen on the *dependency-isolation* axis → split so consumers
  install only what they need.
- **Repo count** is chosen on the *atomic-change* axis → one repo, because we make frequent
  cross-boundary edits ("move RSAM from access to features", "change access in one place")
  and have no external consumers pinning versions. Multi-repo would turn each such edit into
  a multi-release relay and force a version-compatibility matrix.
- Revisit the repo decision later **without touching import structure** if some distribution
  earns an independent release cadence or external consumers.

---

## 8. Namespace packaging (native, PEP 420)

Many small distributions share one import umbrella `aitana.`:

```python
import aitana.access      # from the volcano-access distribution
import aitana.features    # from the volcano-features distribution
import aitana.composition # from the volcano-composition distribution
```

**Mechanism:** omit `__init__.py` at the namespace level. Use native namespace packages
(PEP 420, Python 3.3+) — we're Python 3 + pip only, so ignore pkgutil-style and
pkg_resources-style entirely.

```
aitana-access/                 # distribution
  pyproject.toml
  src/
    aitana/                    # namespace — NO __init__.py here
      access/
        __init__.py             # real package — has __init__.py
        ...
```

**The one gotcha that bites:** if *any* distribution ships `src/aitana/__init__.py`, it
claims the namespace and shadows the others once installed. Leave it out **everywhere**.
Add a CI assertion that no built wheel contains `aitana/__init__.py` at the namespace level.

**Build tool note:** modern setuptools scans implicit namespaces by default via
`[tool.setuptools.packages.find]` in `pyproject.toml` (or use `find_namespace_packages()`
explicitly on older setups). Hatch/uv handle it without ceremony.

Reference docs:
- PyPA guide: https://packaging.python.org/en/latest/guides/packaging-namespace-packages/
- setuptools discovery: https://setuptools.pypa.io/en/latest/userguide/package_discovery.html
- PEP 420: https://peps.python.org/pep-0420/

---

## 9. Packaging & publishing to PyPI

**There is nothing special about uploading a "namespace package."** PyPI only knows
distributions. Each distribution is its own PyPI project, uploaded independently. The
`aitana.` namespace materialises automatically at import time once ≥2 are installed.

- **Naming convention:** distribution `aitana-access` (hyphen) ships import package
  `aitana.access` (dot). Mirrors `azure-storage-blob` → `azure.storage.blob`.
- **No parent project** for `aitana` itself.

### Local flow (per distribution)

```bash
cd aitana-access
python -m build            # → dist/*.whl + dist/*.tar.gz
twine upload dist/*        # TestPyPI first (see below)
```

### Recommended: trusted publishing (OIDC), not API tokens

Use `pypa/gh-action-pypi-publish@release/v1` with GitHub OIDC — no token to manage. If any
old `PYPI_API_TOKEN` secrets exist, remove and revoke them. Use PyPI "pending publisher" to
configure trust *before* the project exists.

Minimal publish job (note the two structural rules in comments):

```yaml
# Build in a SEPARATE job from publish (security: avoids build-dep poisoning /
# privilege escalation under trusted publishing). Call the publish action once per
# job; do NOT invoke from reusable/composite workflows (unsupported).
publish:
  needs: [build]
  runs-on: ubuntu-latest
  environment: pypi
  permissions:
    id-token: write          # MANDATORY for trusted publishing
  steps:
    - uses: actions/download-artifact@v4
      with: { name: python-package-distributions, path: dist/ }
    - uses: pypa/gh-action-pypi-publish@release/v1
```

### Monorepo-specific wrinkles

1. **Per-distribution trusted publisher.** Register one owner/repo/workflow entry on PyPI
   per distribution. Cleanest: one workflow file (or one GitHub Environment) per distribution
   so each PyPI project trusts a distinct workflow.
2. **Publish only what changed.** Use path-filtered triggers or per-distribution tags
   (`features-v1.2.0`, `access-v0.5.1`). Tag convention is usually more predictable for
   deliberate library releases.
3. **Dependency ordering.** When `aitana-composition` depends on `aitana-access`, its
   `pyproject.toml` declares a real versioned dep (`aitana-access>=0.5`). The *published*
   wheel needs that constraint satisfiable on PyPI — so **publish depended-upon distributions
   first**, and never publish a dependent pinned to a version that isn't live yet. (This is
   the mild "relay" cost of splitting; real but small when releases are infrequent.)

### First-publish sequence

Prove the whole pipeline once on **TestPyPI** with a single distribution
(`repository-url: https://test.pypi.org/legacy/`), confirm `pip install` from TestPyPI yields
a working `import aitana.access`, then replicate for the others against real PyPI. This
catches the `__init__.py`-shadowing and dependency-ordering issues before they're frozen into
a real version number you can't reuse.

---

## 10. Open decisions (resolve before/while scaffolding)

These are genuinely yours to decide — Claude Code should ask rather than assume:

1. **Build backend / workspace tool:** uv workspaces vs Hatch vs setuptools+pip. Affects
   lockfile and dev-install ergonomics. (Recommend uv or Hatch for low ceremony.)
2. **Waveform contract type:** lean directly on ObsPy `Stream`/`Trace`, or define a thin
   `contracts` wrapper? Wrapper buys decoupling from ObsPy at the cost of a translation layer.
3. **Does the stored-or-compute split actually apply?** Per §5 collapse rule — list which
   metrics are fetched-only, computed-only, or both. If none are "both", `composition` doesn't
   need the policy layer yet.
4. **Release trigger:** per-distribution tags vs path-filtered `on: release`. (Recommend tags.)
5. **Distribution granularity:** confirm the initial set —
   `aitana-contracts`, `aitana-access`, `aitana-features`, `aitana-composition` — and
   whether consumers ship from this repo or live elsewhere.

---

## 11. Suggested first Claude Code session

1. Scaffold monorepo: `src/`-layout namespace packages for `contracts`, `access`, `features`,
   `composition`; one `pyproject.toml` per distribution; workspace config + single lockfile.
2. Verify the namespace works: install all in editable mode, confirm
   `import aitana.access` / `aitana.features` / `aitana.composition` all resolve.
3. Add **`import-linter`** contracts enforcing the §3 dependency-direction rule
   (`features` must not import `access`; nothing imports upward).
4. Add CI guardrail: assert no built wheel contains `aitana/__init__.py` at namespace level.
5. Move the "combine streams per volcano" logic out of `access` into `composition`.
6. Move RSAM *computation* into `features`; leave stored-RSAM *fetching* in `access`.
7. Stand up the TestPyPI trusted-publishing pipeline for one distribution end-to-end.

> Guardrails first (steps 3–4), then refactors (5–6). The linter makes the boundary
> violations visible, which is usually where the "where's the line?" feeling actually
> comes from.

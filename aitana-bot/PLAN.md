# Aitana Virtual Volcanologist — Duty Officer Companion (v1 scope: Ruapehu)

## Context

A volcano duty officer's job splits into two distinct activities:

1. **Reporting.** GeoNet publishes Volcanic Activity Bulletins (VABs) irregularly, as-needed, when a duty volcanologist decides there's something to report. The long-term goal is to move toward a **daily cadence** — a "volcano weather report" provided every day regardless of whether anything changed, the way a forecast is.
2. **Eruption confirmation.** When monitoring data spikes suddenly, the duty officer has to decide whether an eruption has actually occurred. During daylight hours in good weather this is often a quick webcam check. Otherwise — night, cloud, steam obscuring the crater — it falls back to interpreting seismic and acoustic (infrasound) recordings, which takes more expertise and time under pressure.

This project builds an agent that assists with both: it drafts textual reports by combining volcano monitoring inputs from various APIs (e.g. GeoNet's Tilde API for lake temperature, gas flux, deformation); separately, it helps a duty officer work through "is this an eruption?" by pulling and interpreting webcam imagery and seismo-acoustic data on request.

These are two different problems with two different architectures — see **Capability 1** and **Capability 2** below — bundled under one companion because they serve the same person doing the same job.

**Shared constraint across both capabilities:** this assists an official hazard-response role. Wrong numbers, an invented alert-level assessment, or an unwarranted "yes that's an eruption" have real consequences (aviation, emergency management, public safety). Nothing in this project auto-publishes or auto-escalates — every output is a draft/assessment for a human to act on.

**v1 scope: Ruapehu only, both capabilities.** Eruption confirmation is **on-demand** — the duty officer initiates a check; no continuous/automated monitoring or alerting in this version.

**Reuse first.** `aitana-access` already fetches seismic waveforms (FDSN/S3/SDS), `aitana-features` already computes RSAM/SSAM/DSAR, `aitana-anomaly` already does PCA/autoencoder/scattering-network baseline scoring. Capability 2 is a thin caller on top of these, not a reimplementation. Webcam imagery is fetched from the same `geonet-open-data` S3 bucket that `aitana.access.seismic.S3Waveforms` already reads (anonymous AWS Open Data access) — so the new camera fetcher belongs in `aitana-access` as a sibling of `seismic.py`, not inside the virtual-volcanologist package. It's still new code (nothing reads `camera/` today), just not a new package.

**Package layout.** Ships as distribution `aitana-virtual-volcanologist`, import package `aitana.virtual_volcanologist`, with two sibling subpackages — `aitana.virtual_volcanologist.vab` and `aitana.virtual_volcanologist.eruption_confirm` — consuming `aitana.access` / `aitana.features` / `aitana.anomaly` / `aitana.volcanoes` per the monorepo's layered-architecture rule (imports point downward only).

---

## Capability 1 — Volcanic Activity Bulletins (VAB)

A VAB is an official hazard communication, not a generic report. Wrong numbers or an invented alert-level assessment have real consequences. So the architecture is built around three rules that are non-negotiable, not stylistic preferences:

1. **The agent never infers the Volcanic Alert Level from data.** It's a human decision made at the weekly meeting; the agent transcribes it from parsed meeting notes and carries it forward until the next meeting updates it. This is enforced structurally (the LLM's output schema has no field for it), not just by prompt instruction.
2. **Raw time-series never goes straight into the LLM.** Data is digested into statistics (trend, delta, baseline comparison) first.
3. **Everything produced is a draft.** No auto-publish path exists in this version. A same-day anomaly check (independent of the weekly meeting cycle) flags notable deviations for human review, so a spike on a non-meeting day doesn't silently wait a week.

### Confirmed technical grounding (verified live, 2026-07-13, not assumed)

- **Tilde API** (`https://tilde.geonet.org.nz/v4/`): path pattern `data/{domain}/{station}/{name}/{sensorCode}/{method}/{aspect}/{start}/{end}`, `Accept: text/csv` or `application/json`. `latest/{duration}` only accepts `30d,7d,2d,1d,6h,30m`. Server-side aggregation via `?aggregationPeriod=1d&aggregationFunction=mean|min|max|sum|Perc(N)`.
- **Discovery gotcha (confirmed):** `GET /v4/dataSummary/{domain}?station={code}` works; `/v4/dataSummary/{domain}/{station}` returns 400. Worth a regression test so this doesn't get "fixed" back incorrectly later.
- **Ruapehu series confirmed live via dataSummary:**
  - `envirosensor/RU001/lake-temperature` — sensorCode **01** (1993–2025-11-13, dead) and **04** (2019–2024-08-28, dead) are stale; **06 and 07 are both currently live in parallel** (2025-05-13 → 2026-07-12), each with snapshot/min/max. Pick one as primary, treat the other as a cross-check — don't hardcode a single sensorCode as ground truth.
  - `envirosensor/RU001` also carries lake-height, air-CO2/SO2/H2S-conc (sensorCodes 09/10/11), air-temperature, wind — all live.
  - `gnss/VGMT` — vertical/east/north displacement, method `1d`, live.
  - `scandoas/RUD01` — SO2 gasflux, `automatic` (near-real-time) and `reviewed` (QC'd, ~3wk lag), live.
  - `manualcollect/RU001` — water chemistry (Cl, SO4, pH, Mg, thermocouple temp, gas chemistry), sensorCodes **MC01** and **MC04** currently live (MC03 dead since 2024-03-21); periodic (weeks between samples), not continuous.
  - **Sensor-code churn is a real, observed operational risk** (not hypothetical) — RU001 lake-temperature alone has rotated through 4 sensor codes since 1993, two of them currently active in parallel. The digest layer must check freshness against `dataSummary` rather than trust a hardcoded code.
- **Seismicity/RSAM gap — verify, don't assume.** GeoNet's Tilde paper (Sept 2024) lists domains as coastal/dart/envirosensor/geomag/manualcollect, with RSAM/SSAM explicitly "will also be added" (future work at that time). But domains have since expanded (gnss, scandoas now exist and weren't in that list) — so whether RSAM has shipped by now must be checked live (sweep `dataSummary` for domain names like `seismic`/`rsam`/`tremor`) as the very first task of the relevant milestone, not assumed either way. If it's still absent, seismicity in the report must be sourced qualitatively from meeting notes only — no fabricated quantitative seismic digest. Fallback either way: `aitana-features` can compute RSAM directly from waveforms fetched via `aitana-access` (already used in Capability 2), so a Tilde gap doesn't block quantitative seismicity forever.
- **Style fixtures:** GeoNet's real Ruapehu VABs (fetchable from geonet.org.nz/volcano/vab) follow a consistent skeleton — headline (volcano + VAL + colour code), summary, observations by monitoring method, scientific interpretation, status assessment (increasing/decreasing/stable), issuing signature line. Fetch 2–3 *current* real bulletins fresh when building the fixture set (M5) — don't reuse any bulletin text recalled from a prior session without re-fetching, given attribution/exact wording can't be assumed accurate from memory.

### Project structure

```
aitana-virtual-volcanologist/
├── pyproject.toml
├── data/
│   ├── state/ruapehu.json       # persisted current VAL/colour/meeting context
│   ├── thresholds/ruapehu.yaml  # domain-expert-editable background ranges
│   ├── meeting_notes/            # inbox: user drops .md/.txt weekly notes here
│   └── fixtures/
│       ├── bulletins/            # real current GeoNet bulletins, style few-shot
│       ├── tilde_cassettes/      # recorded API responses for offline tests
│       └── meeting_notes/        # sample notes for tests
├── reports/                      # output: reports/ruapehu-2026-07-13.md
├── src/aitana/virtual_volcanologist/
│   └── vab/
│       ├── cli.py                 # Typer: digest / ingest-notes / check-anomalies / draft / run
│       ├── config.py
│       ├── models.py              # all Pydantic schemas
│       ├── volcanoes/ruapehu.py   # station/sensor registry (absorbs sensor-code churn)
│       ├── digest.py              # raw series → DigestStat, incl. freshness check
│       ├── thresholds.py          # YAML → ParameterThreshold; DigestStat → AnomalyFlag
│       ├── meeting_notes.py       # notes file → VolcanoState update (extraction, not inference)
│       ├── state.py               # load/save VolcanoState JSON
│       ├── style_fixtures.py      # loads bulletin fixtures for few-shot prompt
│       ├── compose.py             # one Anthropic API call → structured VabDraftLLM
│       ├── template.py            # VabDraft → rendered Markdown
│       └── verify.py              # provenance check: every number traces to source data
└── tests/
    ├── unit/  (test_digest, test_thresholds, test_meeting_notes, test_verify, test_template)
    └── integration/test_cli_dry_run.py
```

Deps: `httpx`, `pydantic>=2`, `pyyaml`, `anthropic`, `typer`, `rich`, `pytest`.

### Data model (`models.py`)

The key structural decision: **`VabDraftLLM` (what Claude produces) has no field for Volcanic Alert Level or Aviation Colour Code.** `compose.py` gets `VabDraftLLM` back from the model, then deterministically merges in `VolcanoState.volcanic_alert_level`/`.aviation_colour_code` to build the final `VabDraft`. The model structurally cannot emit a level even if it tried.

- `TildeSeriesKey` — domain/station/name/sensor_code/method/aspect.
- `DigestStat` — parameter, series_key, unit, window, n_obs, latest/min/max/mean, delta_vs_previous_period, trend, baseline comparison, `data_quality_note` (for the freshness/sensor-churn check).
- `ParameterThreshold` — background_min/max, max_period_delta, baseline_method (`fixed` for v1; `rolling_window` reserved for later), notes (domain expert's rationale, free text).
- `AnomalyFlag` — parameter, severity (`notable`/`urgent`), description, recommend_review.
- `VolcanoState` — volcano, volcanic_alert_level, aviation_colour_code, meeting_date, key_context, `raw_excerpt` (verbatim substring of the source note — traceability guarantee).
- `ObservationClaim` — text, parameter, cited_value, cited_unit (every number the LLM writes must cite one of these, tied to a real `DigestStat.parameter`).
- `VabDraftLLM` → `VabDraft` (adds volcano/level/colour/anomaly_flags, server-assembled).
- `NumberCheck` / `VerificationReport` — cross-check every `ObservationClaim.cited_value` against the actual `DigestStat` it claims to cite.

### LLM integration

**Anthropic Python SDK directly, not the Claude Agent SDK.** This is a linear pipeline — Python code fetches and digests structured data, then makes **one** Claude call to turn already-retrieved data into a structured draft. No open-ended exploration or tool loop is needed, so the Agent SDK (built for autonomous multi-turn tool use) would be unnecessary overhead here. (Contrast with Capability 2, where the investigative nature of the task is exactly why the Agent SDK earns its keep.)

- Use structured output (`client.messages.parse(..., output_format=VabDraftLLM)`) so the response is validated straight into the Pydantic model — no free-text number generation to trust.
- System prompt encodes the three non-negotiable rules directly (drafting not deciding; every number must cite a `DigestStat` claim; draft language only, e.g. "draft observations suggest" not "GeoNet confirms"), plus 2–3 real current bulletins as few-shot structure/tone reference.
- **Model choice — a cost/quality call for you, not fixed by default policy:** since this pipeline re-verifies every number regardless of which model wrote it (`verify.py`), start with `claude-sonnet-5` given the target is daily-cadence (365 runs/year) — reserve `claude-opus-4-8` as an explicit A/B comparison during the pilot if narrative quality noticeably matters, not as the default.
- **Prompt caching**: the system prompt (rules + fixtures) is identical every run; only the day's digest/state changes. Mark the system block `cache_control: ephemeral` — at daily cadence this meaningfully cuts cost.

### Anomaly/threshold design (`thresholds.py` + `data/thresholds/ruapehu.yaml`)

Per-parameter YAML, editable without a code change: `background_min`/`background_max`, `max_period_delta`, `baseline_method` (fixed for v1), free-text `notes` for the domain expert's rationale. Flag `notable` if latest value is outside background range or period delta exceeds the rate-of-change guard. Keep escalation conservative — the system's entire job is to flag for a human, never to auto-decide. Seed initial thresholds from whatever background ranges the volcano team already uses operationally (don't invent numbers) — this needs a real conversation with the duty volcanologist, not values guessed during this planning session.

### Testing strategy

- No network in unit tests: `TildeClient` takes an injectable `Fetcher`; tests use a `FixtureFetcher` reading recorded JSON/CSV cassettes.
- `digest`, `thresholds`, `verify`, `template` are pure functions over Pydantic models — test with synthetic data, including a deliberately-wrong-claim case in `test_verify.py` to confirm the check actually catches fabricated numbers, not just passes on good input.
- `meeting_notes.py` tested against a few hand-written sample notes (level unchanged / level changed / ambiguous) asserting `raw_excerpt` is a real substring of the source file.
- `compose.py` is the only piece touching the live API — a `--dry-run` mode swaps in a canned `VabDraftLLM` so everything downstream (template, verify, CLI) is testable with zero API cost.

### Milestones (each independently runnable/demoable)

1. **M1 — Tilde client + envirosensor digest.** `tilde_client.py`, `digest.py`, `volcanoes/ruapehu.py` with confirmed RU001 series (lake-temp, gas). First task: live-check current sensorCode(s) for lake-temperature (06/07 both active as of this planning session — reconfirm) and sweep for a seismic/RSAM domain. *Verify:* `vab digest --volcano ruapehu --window 7d` prints stats; hand-check 2–3 values against a manual `curl`.
2. **M2 — Add GNSS, scanDOAS, manualcollect chemistry.** *Verify:* `vab digest --volcano ruapehu --window 30d` shows all domains.
3. **M3 — Meeting-notes parser + persisted state.** *Verify:* `vab ingest-notes <file>` then `vab state show --volcano ruapehu` — VAL/colour/date match the note, `raw_excerpt` is a real quote.
4. **M4 — Threshold config + anomaly flagging** (independent of the LLM). *Verify:* `vab check-anomalies --volcano ruapehu`; temporarily tighten a threshold in YAML, confirm a flag appears, revert.
5. **M5 — LLM composition + Markdown rendering.** Fetch 2–3 *current* real Ruapehu bulletins fresh for the fixture set. *Verify:* `vab draft --volcano ruapehu` writes `reports/ruapehu-YYYY-MM-DD.md`; VAL/colour in the output must match `state/ruapehu.json` exactly; a "DRAFT — NOT FOR PUBLICATION" banner is present.
6. **M6 — Provenance verification.** *Verify:* real run passes; a unit test with a hand-fabricated wrong claim value fails as expected.
7. **M7 — Full orchestration + offline mode.** `vab run --volcano ruapehu` (live) and `--dry-run` (fixtures only, zero cost/network) — the dry-run mode is what CI uses and what a future daily cron would eventually wrap.

### Open items to resolve with the volcano team before/during implementation

- Real background/threshold values per parameter (seed values must come from the duty volcanologist, not guessed).
- Whether reusing real GeoNet bulletin text as style fixtures needs a licensing check, even for an internal GNS tool.
- Confirm live seismic/RSAM domain availability in Tilde (see M1) — determines whether seismicity is digested from Tilde directly or via the `aitana-features` RSAM-from-waveforms fallback.

---

## Capability 2 — Eruption Confirmation

### Problem

A duty officer sees a sudden spike (in RSAM, in a Tilde parameter, in a public report) and needs to work out, quickly, whether it's an eruption. In good daylight weather this is often a webcam glance. Otherwise it means pulling seismic and acoustic (infrasound) recordings and interpreting them — the harder, more expertise-dependent path this capability targets. **v1: Ruapehu only, on-demand — the duty officer initiates a check, nothing runs continuously or alerts on its own.**

### Non-negotiable rules

1. **The agent never asserts that an eruption occurred.** That determination — and any consequent action (VAL change, aviation colour code, emergency management activation) — is a human call. The agent's output is a structured, evidence-backed *assessment* ("what the data show, and how strongly"), not a verdict. Enforced structurally the same way as Capability 1's VAL rule: the output schema has no `eruption_confirmed: bool` field — only an evidence list plus a hedged, human-reviewable characterization (e.g. `"consistent with eruptive activity"` / `"consistent with background"` / `"inconclusive — insufficient data"`).
2. **Every claim must trace to a specific tool-fetched artifact.** For seismo-acoustic data this means the same digest-before-LLM discipline as Capability 1 (RSAM/SSAM/anomaly-score values, not raw waveform arrays, go into the prompt). For webcam imagery the discipline adapts rather than disappears: interpreting an image *is* the task, so the raw frame legitimately goes to the model — but the tool must return a specific, timestamped frame (never "live video"), and any visual claim in the output must cite that exact camera ID + timestamp, so a reviewer can pull up the same frame and check the agent's read of it.
3. **All tools are read-only; every run produces a draft for human review, and the full tool-call transcript is logged.** The agent can fetch and compute; it cannot publish, alert, or write to `VolcanoState`. Because this is a multi-step agentic session rather than one auditable LLM call (contrast Capability 1), logging exactly what was fetched and in what order is what makes the final assessment reviewable after the fact — this is not optional tidiness, it's the audit trail a real incident review would need.

### Architecture — why the Agent SDK here, and not a linear call

Capability 1's "is this a spike" question is answered by a fixed, known set of statistics — a linear pipeline (digest → one LLM call) is the right shape, and its auditability is a feature. Capability 2's "is this an eruption" question is open-ended: what to check next depends on what was just seen (webcam is clear and shows a plume → done; webcam is clouded out → fall back to seismicity; seismicity is elevated but ambiguous → check infrasound/acoustic for the airwave signature that distinguishes an explosive eruption from a large earthquake or noise; compare against an anomaly baseline to see how unusual this really is relative to background). That branching is exactly what a fixed pipeline can't do and an agentic tool loop can, so this capability uses the **Claude Agent SDK** with a small set of read-only tools, not the raw Anthropic SDK.

### Confirmed technical grounding — webcam imagery (verified live, 2026-07-13)

- **Bucket and layout confirmed** via GeoNet's AWS Open Data docs (geonet.org.nz/data/access/aws) and the `GeoNet/data-tutorials` repo: images live at `s3://geonet-open-data/camera/{building|landslide|volcano}/...`. Volcano monitoring cameras are under `camera/volcano/`.
- **Key pattern:** `camera/volcano/images/{yyyy}/{SITE}/{SITE}.{VC}/{yyyy}.{doy}/{yyyy}.{doy}.{hhmm}.{ss}.{SITE}.{VC}.jpg` — 4-char site code, 2-char view code, JPG, captured at **10-minute intervals**.
- **Same access pattern as seismic data, already implemented:** `aitana.access.seismic.S3Waveforms` already does anonymous `fsspec`/S3 access (`target_options={"anon": True}`) against this exact `geonet-open-data` bucket for miniseed. A new `aitana.access.camera` module can mirror that class almost directly — swap the miniseed key template for the camera key template above.
- **Unresolved and load-bearing for this capability specifically:** GeoNet's general AWS Open Data page states bucket sync is **daily** and recommends FDSN/Tilde instead "for near-real time access." If that daily-sync caveat also applies to `camera/volcano/`, the S3 archive is unusable for an *on-demand, right-now* eruption check (it would show yesterday's crater, not this morning's) — it would still be fine for the E5 fixture/replay set (retrospective, not live). **Do not assume either way** — the first task of E1 is checking actual object timestamps in `camera/volcano/` against wall-clock time to see whether new frames really do land within ~10–20 minutes, or whether a different, genuinely live source (e.g. a live image URL on geonet.org.nz itself) is needed for the on-demand tool. This is a repeat of the same "verify, don't assume" pattern Capability 1 applied to the Tilde RSAM gap.
- **Not yet confirmed:** which `SITE`/`VC` codes correspond to Ruapehu's crater-facing camera(s) — needs listing `camera/volcano/images/{current yyyy}/` or asking the team.

### Tools exposed to the agent

All tools are thin wrappers over existing or sibling `aitana-access`/`aitana-features`/`aitana-anomaly` code — no algorithm work needed, just exposing already-built (or near-identical-pattern) functionality as Agent SDK tools:

- `fetch_webcam_frame(camera_id, at="latest")` — wraps a new `aitana.access.camera` module (see grounding above), living in `aitana-access` alongside `seismic.py` since both are pure I/O against the same S3 bucket. Blocked on confirming real-time latency and Ruapehu's site/view codes (see open items).
- `fetch_seismic_window(station, channel, start, end)` — wraps `aitana.access.seismic.SeismicData` (FDSN/S3/SDS clients, already exists).
- `compute_rsam(...)`, `compute_ssam(...)`, `compute_dsar(...)` — wrap `aitana.features` (already exists).
- `check_anomaly(...)` — wraps `aitana.anomaly` (PCA/autoencoder/scattering-network baseline comparison), *if* a trained baseline already exists for the relevant Ruapehu station(s) — otherwise this tool is blocked on training one (see open items).
- `get_recent_digest(...)` — reuses Capability 1's `digest.py`/Tilde client, since a real Ruapehu eruption often has a crater-lake signature (temperature/level) that corroborates or contradicts the seismo-acoustic read.

### Data model additions

- `WebcamFrameRef` — camera_id, timestamp, image reference.
- `EruptionEvidence` — kind (`visual`/`seismic`/`acoustic`/`geochemical`), description, cited artifact (a `WebcamFrameRef` or a digest/feature stat) — the Capability-2 analogue of Capability 1's `ObservationClaim`.
- `EruptionAssessment` — list of `EruptionEvidence`, a hedged `characterization` (no boolean verdict field, per rule 1), confidence, caveats (e.g. "webcam obscured by cloud, assessment based on seismo-acoustic only").
- `ToolCallLog` — full record of what was fetched/computed and when, for the audit trail (rule 3).
- `verify.py` (Capability 2) — checks every `EruptionEvidence.cited_artifact` traces to a real entry in the `ToolCallLog`, the agentic-loop analogue of Capability 1's provenance check.

### Project structure

```
aitana-access/src/aitana/access/
└── camera.py        # NEW — S3Camera, mirrors S3Waveforms in seismic.py: anonymous
                      # fsspec/S3 access to geonet-open-data, camera/volcano/ key template

aitana-virtual-volcanologist/src/aitana/virtual_volcanologist/
└── eruption_confirm/
    ├── tools.py      # Agent SDK tool defs: fetch_webcam_frame, fetch_seismic_window,
    │                 # compute_rsam/ssam/dsar, check_anomaly, get_recent_digest
    ├── models.py     # WebcamFrameRef, EruptionEvidence, EruptionAssessment, ToolCallLog
    ├── agent.py      # Claude Agent SDK session wiring + system prompt (rules above)
    ├── verify.py     # citation check: every EruptionEvidence traces to a real ToolCallLog entry
    └── cli.py        # `vv confirm --volcano ruapehu` (interactive, on-demand)
```

`fetch_webcam_frame` in `tools.py` just calls `aitana.access.camera.S3Camera`, matching Design.md's rule that `access` owns all I/O and everything above it consumes rather than re-implements.

### Testing strategy

- Fixture webcam frames (clear/plume, clear/no plume, night, cloud-obscured) and fixture waveform cassettes — check first whether `aitana-access`/`aitana-features` already have a cassette pattern from their own test suites before building a new one.
- Fake/injectable tool implementations so unit tests hit zero live network/API cost, mirroring Capability 1's `FixtureFetcher` pattern.
- A "no absolute claims" test: scan every generated `EruptionAssessment` for forbidden verdict language ("confirmed", "eruption has occurred", "is erupting") — the Capability-2 analogue of Capability 1's fabricated-claim test in `test_verify.py`.
- Replay of known past events (see E5) graded by the volcano team — this is a human-judgment gate, not something CI can certify alone, mirroring how Capability 1's thresholds need real domain-expert seeding.

### Milestones (E1–E5, additive to M1–M7 above; independently runnable/demoable)

1. **E1 — Webcam tool.** First task: verify whether `camera/volcano/` object timestamps in `s3://geonet-open-data` actually land close to wall-clock time (not the daily-lag the general AWS Open Data docs warn about), and identify Ruapehu's `SITE`/`VC` code(s). Then build `aitana.access.camera.S3Camera` (mirroring `S3Waveforms`) and the `fetch_webcam_frame` tool, test against 2–3 saved sample frames spanning clear/cloudy/night. *Verify:* tool returns a real image + accurate timestamp for a live request; if latency turns out to be ~daily, this milestone's exit criterion changes to documenting that constraint and deciding a fallback with the team rather than shipping a tool that silently returns stale frames.
2. **E2 — Seismo-acoustic tools.** Thin Agent SDK wrappers over `aitana.access.seismic` + `aitana.features` (RSAM/SSAM/DSAR) — no new algorithms, just tool-ify existing library calls. *Verify:* a tool call against a known time window returns an RSAM value matching a hand-check against `aitana-features`' own tests.
3. **E3 — Anomaly scoring tool.** Wrap `aitana.anomaly` baseline comparison; confirm whether a trained Ruapehu baseline already exists or needs training first (open item). *Verify:* tool flags a known historically-anomalous window as anomalous and a known quiet window as not.
4. **E4 — Agent loop + cited assessment.** Wire E1–E3 tools into the Claude Agent SDK; enforce the evidence-citation schema and the forbidden-verdict-language check. *Verify:* `vv confirm --volcano ruapehu` runs an on-demand session, full tool-call transcript is logged, output never asserts an eruption as settled fact.
5. **E5 — Duty-officer review harness.** Replay past known events (at least one real spike and, if usable data exists, a real historical Ruapehu eruption/lahar-triggering event — confirm with the team which cases have data) through fixtures; the volcano team manually grades the assessments before any live use. *Verify:* team sign-off recorded — this milestone's success criterion is human judgment, not a passing test.

### Open items to resolve with the volcano team before/during implementation

- **Webcam real-time latency.** Path structure is confirmed (`s3://geonet-open-data/camera/volcano/...`, see grounding above), but GeoNet's own AWS Open Data docs warn general bucket sync is daily and point to FDSN/Tilde for near-real-time instead — unclear if that applies to camera imagery specifically, given images are captured every 10 minutes at the source. Must be checked live (compare object timestamps to wall-clock) before E1 is considered done; if it really is ~daily, this tool is unusable for on-demand confirmation and a different live-image source needs discussing with the team.
- **Ruapehu camera site/view codes.** Which `SITE`/`VC` codes under `camera/volcano/` correspond to Ruapehu's crater-facing camera(s) — needs listing the bucket or asking the team.
- **Anomaly baseline availability.** Does `aitana-anomaly` already have a trained baseline (PCA/autoencoder) for the relevant Ruapehu seismic station(s), or does E3 need to train one from scratch first?
- **Station/channel selection.** Which seismic and (if available) infrasound/acoustic stations and channel codes are actually the relevant ones for Ruapehu eruption confirmation — needs domain-expert input, not a guess from station lists.
- **What "confirmed" means operationally.** What criteria does the duty officer currently use in practice to call an eruption from seismo-acoustic data alone (no webcam)? The `EruptionAssessment.characterization` taxonomy should mirror real decision practice, not an invented one.

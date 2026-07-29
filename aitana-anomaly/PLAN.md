# aitana-anomaly — State & Anomaly Detection Benchmarking Roadmap

If we think of a volcano as a dynamical system, we can assign a theoretical, hidden state vector s(t) to it. Assuming that volcanoes go through certain state regimes repeatedly, it could be very helpful for volcanic unrest detection to determine the current regime. Automatically detecting changes in the state has the potential to be very useful for anomaly detection. 

## Context

 A state-detection method segments the time series directly into a sequence of discrete, recurring states with a transition structure — "normal vs. not-normal" and "this looks like the state we were in during May 2022" both become direct queries against that state sequence.

The initial goal is to implement 4-5 algorithms for state detection with increasing complexity and compare their results for two observational periods: Ruapehu 2021-01-01 to 2022-12-31 and Whakaari 2018-01-01 to 2020-03-01 using all observational data that is collected continuously (continuous seismic data, temperature, gas, water level, earthquakes). 

Our baseline algorithm is using Principal component analysis to generate an orthogonal space of a suite of seismic features as well as earthquake rates and gas flux. We then use a Hidden Markov Model to learn underlying states of the system.

The next level of complexity is a 1D-convolutional autoencoder over a sliding window of timesteps and the Scattering Network.

Finally, the most complex methods are the scattering network at the clasp algorithm.

For historical reasons, delay-embedding could be used as a 5th algorithm.

Beyond the initial evaluation phase and a peer-reviewed manuscript we would like to run several state detection algorithms operationally to form an **expert panel**, detect anomalies automatically and then provide human experts with an interpretable picture of that anomaly through their underlying state detections.

## Implementation
Validation runs both synthetically (fault injection, ground truth known — repurposed as a diagnostic of which expert catches which fault type, not a ranking mechanism) and against real eruption and unrest catalogs (Whakaari and Ruapehu), integrated with `aitana-bench`. This is a multi-month research roadmap, not a single implementation task — phased so the harness gets validated end-to-end on cheap wins (AE, PCA) before the CLaP integration and the full expert-panel synthesis.

`aitana-anomaly` currently contains three disconnected, mostly-incomplete pieces: a working but unscored Autoencoder, a fully broken PCA module (its test suite is entirely skipped), and a scattering-network feature extractor with no downstream consumer. None of them implement `compute_anomaly_index()`, the scalar `[0,1]` score the base class promises — so today nothing in the package actually produces a usable state detector or anomaly signal, and nothing is benchmarked.

`claspy` (the ClaSP / CLaP family, Ermshaus, Schäfer & Leser) provides a peer-reviewed and maintained method time-series segmentation: `BinaryClaSPSegmentation` for change-point detection and `state_detection.AgglomerativeCLaPDetection` for recurring-regime identification with consistent labels and recovered transition structure across time — published at VLDB 2025, BSD-3-licensed, actively maintained, handles multivariate input, needs no pre-specified state count, with prior applications to multivariate sensor data (accelerometer/magnetometer, ECG).


- **Algorithms**: PCA + HMM, Autoencoder (extend existing working code, moving from a per-timestep dense architecture to a windowed 1D-convolutional one — see below), Scattering Network (add a new novelty-detection scoring layer), and ClaSPy/CLaP-based state detection — four experts in an ensemble panel, ordered by increasing modeling complexity, none designated as "the" method. 

- **Validation**: dual-track — synthetic fault injection plus real-world scoring against known eruptions via `aitana-bench`. Fault injection's role is a **per-fault-type sensitivity diagnostic**: for each injected fault type/magnitude, record which algorithms' alarm bits fire, producing a fault-type × algorithm characterization matrix rather than an FP/FN ranking. Real-eruption AUC scoring stays as useful per-algorithm context but isn't combined into a ranked table either.

- **Agreement signal** (the core of the expert-panel framing, layered rather than a single metric):
  1. **Vote count** — threshold each algorithm's own `[0,1]` index against its own quiet-period percentile (e.g. 95th) to get a binary alarm bit per algorithm per timestep, summed to a 0–4 vote count. Cheap, robust to differing scales, and the literal "voting system."
  2. **Consensus strength** — rolling-window rank correlation across the four `[0,1]` indices, distinguishing "algorithms happened to cross threshold together" from "algorithms are genuinely moving in sync."
  3. **state label as context** — plot inferred states as a colored strip against the eruption/unrest chronology and see if state transitions precede known activity changes.

- **Datasets**: both Whakaari and Ruapehu. Both already have parity in `aitana-volcanoes` (`eruptions()` + `load_all()` in `whakaari.py` and `ruapehu.py`). Ruapehu also has raw 3-component waveform access via `S3Waveforms` (snet.py's own default example stream, `NZ.MAVZ.10.HH?`, is a Ruapehu station), so the Scattering Network can run on both volcanoes too. CLaP is general-purpose and applies to both volcanoes without a signal-type precondition. For PCA+HMM use the same seismic features as the current autoencoder implementation but add additional data streams available through the load_all functions of `aitana-volcanoes`.

- **Real-data validation windows**: Whakaari and Ruapehu are validated asymmetrically because their eruption catalogs differ, not identically. **Whakaari** carries the headline quantitative window, 2018–2020, containing the fatal 2019-12-09 eruption (`Activity_Scale 3` in `WhiteIs_Eruption_Catalogue.csv`) and preceded by documented precursory unrest (2019-04-15 earthquake swarm, per `aitana-bench/src/aitana/bench/visualise.py`'s `trellis_plot` annotations) — a genuine quiet → precursor → eruption arc, scored via `aitana-bench.scoring`'s existing pew-window ROC/AUC machinery against the full catalog (2018–2020 is the reported headline window, not a separate computation slice). **Ruapehu** has no eruption in recent decades to anchor that machinery — GeoNet's live eruption catalog (`ruapehu.eruptions()`) ends 2009-07-13 — so its headline real-data validation is instead a qualitative case study of the domain-expert-confirmed H1 2022 volcanic unrest episode (not encoded in any catalog/CSV today), using full local RSAM/lake/gas coverage for 2021–2022 (2021 as lead-up context), checked against whether the anomaly index rises and/or CLaP flags a distinct/rare state during that period. This is the concrete real-world exercise of Figure 3's `nearest_historical_match()` case study, not a downgrade of Ruapehu's role.

- **New workflow location**: self-contained inside `aitana-anomaly` (`src/aitana/anomaly/workflows/` + `data/workflows/`), registered via `aitana-anomaly`'s own `pyproject.toml` entry points — keeps model code and its benchmark workflow together. Each methodology gets its own workflow.

- **Score normalization**: a shared ECDF/percentile-rank normalizer in `AnomalyDetectionBaseClass`, applied to each baseline's raw statistic (PCA+HMM log-likelihoods and PCA reconstruction residual, AE's reconstruction MSE, SNet's novelty score) against a reference "quiet period" distribution — nonparametric, scale-invariant, comparable across algorithms. CLaP's change-point score profile is already a `[0,1]` "probability of a change point" and skips this step; a separate state-rarity index (see below) is used for its state-level anomaly framing. The vote-count alarm-bit threshold (above) is a separate step layered on top of this normalized `[0,1]` index, not a replacement for it.

- **Fault injection**: corrupts data at the plain DataFrame/array level, before any `Storage`/`Dataset` wrapping — keeps the harness decoupled from `aitana-storage` internals; each algorithm's own glue code re-wraps corrupted data into its native input format.

- **PCA + HMM output**: report both negative log-likelihood and PCA reconstruction residual side by side in the comparison tables rather than forcing a single combined index.

- **SNet novelty detector**: start with a one-class SVM on order1 scattering coefficients (simplest, well-understood default) — revisit if it underperforms once real data is in hand.

- **Cross-algorithm cadence alignment**: All data should be down/up-sampled to 10 min intervals. This will be implemented in `aitana-access` and `aitana-features`.

- **CLaP is batch, not streaming**: `BinaryClaSPSegmentation` and `AgglomerativeCLaPDetection` fit over the whole input series at once, so a score at time *t* can be informed by data at *t' > t* — a retrospective algorithm, unlike PCA/AE/SNet, which each score a new point against an already-fixed reference model. That's the right fit for the retrospective case studies (Figure 3) but means CLaP's track is reported without a detection-latency number (Manuscript-oriented outputs) rather than forcing one. `claspy` does ship an online `streaming.StreamingClaSPSegmentation` ("ClaSS") variant — noted here as a possible future extension for the operational phase.

- **Coupled fault injection for panel-value evidence**: alongside the existing independent per-stream Bernoulli injection, add a coupled/system-wide fault mode — the same fault onset injected simultaneously across a group of correlated streams, calibrated to look like a genuine multi-stream precursor rather than a single sensor glitch (the direct synthetic analogue of `DBN.md`'s coupled-vs-uncoupled section). This gives a quantified test of what the expert panel actually adds: does `vote_count`/`consensus_strength` separate coupled, precursor-like faults from independent, glitch-like faults better than any single algorithm's alarm bit? Without this, the panel's added value is only shown via two real-world N=1 case studies (Whakaari 2019, Ruapehu 2022) — this closes that gap synthetically, reusing the Phase 0 harness rather than adding new machinery.

## Common algorithm contract

Add a shared normalizer to `AnomalyDetectionBaseClass` (`aitana-anomaly/src/aitana/anomaly/anomaly_base.py`):

```python
def normalize_score(raw_scores, reference_scores):
    # ECDF / percentile rank of raw_scores against a quiet reference-period distribution
```

Each baseline subclass's `compute_anomaly_index()` calls this on its own raw statistic (log-likelihood and reconstruction residual for PCA+HMM, per-timestep reconstruction MSE for AE — `autoencoder_loss`, recomputed under the windowed conv-autoencoder architecture from Phase 1 rather than the current per-timestep one, novelty-detector score for SNet). The CLaP-based state detector produces two complementary signals instead of a single reconstruction-error score:
1. **Change-point index**: ClaSP's own score profile, already `[0,1]`, usable directly as `compute_anomaly_index()` for the "is something changing right now" question.
2. **State-rarity index**: for a point assigned to state *k*, `1 - (historical occupancy of state k / total quiet-period length)` — rare/novel states score high, common recurring states score low. This is the direct analogue of reconstruction error for a state-based method, and it, too, needs no ECDF step (already bounded).

A design note: AE and SNet currently consume `Storage`-backed netCDF feature streams via `ZizouDataset`; PCA + HMM (post-rewrite) and the CLaP state detector should also consume data from `Storage`. This will have to be implemented in `aitana-access`

**Future agreement utilities** (Phase 6, not implemented until all four algorithms exist): `alarm_bit(index, reference_scores, percentile=95)`, `vote_count(indices: dict[str, array])`, and `consensus_strength(indices: dict[str, array], window)` consume the per-algorithm `[0,1]` indices described above, already resampled onto a shared cadence.

## Phased roadmap


**Phase 0 — Contract & harness scaffolding** (small)
- Add `normalize_score()` to `anomaly_base.py`.
- New `aitana-anomaly/src/aitana/anomaly/evaluation.py`: `inject_faults(df, columns=None, rate=0.06, offset_range="hard"|"easy"|(k1,k2), mode="independent"|"coupled", coupled_groups=None, rng=None) -> (corrupted_df, mask_df)` — `mode="independent"` is the existing per-stream Bernoulli injection (offset calibrated to `df[col].diff().std()`, hard/easy presets); `mode="coupled"` injects the same fault onset simultaneously across a specified group of correlated streams (`coupled_groups`, e.g. co-located seismic + gas channels), mimicking a genuine multi-stream precursor rather than a single-sensor glitch — the direct synthetic analogue of `DBN.md`'s coupled-vs-uncoupled section, and the mechanism Phase 6 uses to quantify what the expert panel adds beyond any single algorithm (see Decisions made). `score_detector(anomaly_index, mask, threshold)` — thin wrapper around `sklearn.metrics.roc_curve`/`roc_auc_score` against the boolean ground-truth mask (plain pointwise classification, not `aitana-bench.scoring`'s eruption-window apportionment logic — don't reuse that machinery here). Its output feeds the Table 1 sensitivity-profile matrix (Phase 6) and the coupled-vs-independent panel-value diagnostic — the injection mechanism itself is unchanged for the independent mode, only how its output is consumed downstream, plus the new coupled mode.
- New `tests/test_fault_injection.py`.
- *Verify*: inject synthetic faults into a toy DataFrame, recover them with a trivial threshold-on-raw-value detector, confirm the harness's ground-truth bookkeeping is correct independent of any real algorithm; confirm coupled-mode injection lands on the same timestamps across the specified group of columns.

**Phase 1 — Autoencoder anomaly index + temporal receptive field** (medium)
- Replace `TiedAutoEncoder`'s per-timestep dense architecture with a 1D-convolutional autoencoder over a sliding window of consecutive timesteps (encoder: strided 1D convs over the time axis; decoder: transposed convs reconstructing the window) — this is what gives the panel its temporal-dynamics rung now that the DBN is gone (see Context/Decisions). Requires a windowed variant of `ZizouDataset` (`anomaly_base.py`) yielding overlapping `(window_length, nfeatures)` samples instead of single-timestep vectors; window length and stride are a tuning choice, not fixed in advance. This is already implemented in TonikDataset in the aitana-storage package. Start on the order of the existing `stack_dict`'s finest resolution (1h) times a handful of steps, and check sensitivity during this phase's verify step rather than guessing a value up front.
- Reconstruction error is now per-window; attribute it back to individual timesteps (e.g. average error across all windows covering timestep *t*) to produce a per-timestep `autoencoder_loss` compatible with the rest of the contract.
- Implement `AutoEncoder.compute_anomaly_index()` in `autoencoder.py` from this windowed `autoencoder_loss`, using `normalize_score()`.
- *Verify*: extend `tests/test_autoencoder.py` — `compute_anomaly_index()` returns values in `[0,1]`; injected faults on the existing `setup_ac` fixture produce a measurably higher index than clean data; additionally confirm the windowed model catches a slow-onset synthetic drift (a fault type a purely pointwise model would miss) better than the pointwise reconstruction it replaces, as a direct check that the architecture change bought something.

**Phase 2 — PCA rewrite** (medium)
- Rewrite `aitana-anomaly/src/aitana/anomaly/pca.py` to also include a Hidden Markov Model that effectively propagates clusters in the PCA over time. Use the  `hmmlearn` package on the `load_all()`-DataFrame design. Implement negative log-likelihood and PCA reconstruction residual as the two raw statistics, each passed through `normalize_score()`. Remove the dead `runPca`/`runPcaCore` CLI functions that reference undefined `VolcanoMetadata`/`FeatureRequest`.
- *Verify*: replace `tests/test_pca.py`'s `@unittest.skip(...)` tests with new ones (a synthetic multivariate-normal fixture, similar to what the old skipped tests already sketch) confirming known-injected outlier points are flagged by anomaly statistics.

**Phase 3 — Close the loop end-to-end (AE + PCA/HMM only)** (medium)
- In `aitana-bench/src/aitana/bench/scoring.py`: add an injectable `eruptions_fn: Callable = whakaari.eruptions` parameter to `get_evaluation_windows()`, threaded through `compute_rates()`; drop the currently-broken `end_date=` kwarg passed to `whakaari.eruptions()` at line 28 (that function's real
signature is `eruptions(min_size=0, dec_interval=None)` — no `end_date` param exists, so this call raises `TypeError` today regardless of volcano; the
existing `.loc[starttime:endtime]` filter on the next line already makes it redundant). Update the one call site. This is a small, scoped patch for
`get_evaluation_windows()`/`compute_rates()` only.
`evaluate_threshold()` is *not* volcano-agnostic — it hardcodes two Whakaari-derived date literals (`starttime` floored at `"2013-01-01"`; a `"2020-01-09"` cutoff, commented `# as the last explosive eruption was on 09/12/2019`, used to backfill true-negative days). Per the simplicity-first rule, don't generalize this: Ruapehu has no cataloged eruption for these dates to mean anything, so there's no real scenario where Ruapehu needs `evaluate_threshold()`'s catalog-ROC path. Keep it Whakaari-only by design, and add a docstring note saying so ("assumes Whakaari's eruption history; do not call for Ruapehu"). Ruapehu's Phase 5 case-study validation bypasses `evaluate_threshold()`/`get_roc_curve()`/`compute_auc()` entirely, consuming the raw anomaly-index/state-label time series directly rather than pew-window apportionment. Note also: `ruapehu.load_all()`'s bare signature has no `start_date` parameter (unlike `whakaari.load_all()`), but `@cache_dataframe()` (`aitana-access/src/aitana/access/util.py`) accepts and applies `start_date`/`end_date` at the wrapper level regardless of the inner function's own signature, slicing the result via `.loc[]` post-hoc — callers can already pass `start_date="2021-01-01"` today with no change to `ruapehu.py`.
- Add `compute_auc(tpr, fpr)` to `scoring.py`: `make_strictly_increasing()` (already present) on `fpr`, then `np.trapz(tpr, fpr)` — `get_roc_curve()` forces ROC endpoints to `(1,1)` "so AUC is comparable" but never actually computes one today.
- In `aitana-anomaly/pyproject.toml`: add `aitana-bench` dependency, `[tool.uv.sources]` entry, `[project.entry-points."volcanobench.workflows"]` registering `autoencoder` and `pca_hmm`, and `[tool.hatch.build] include = [...]` for the new workflow data dir (mirroring `aitana-bench/pyproject.toml`'s existing patterns).
- New `aitana-anomaly/src/aitana/anomaly/workflows/{__init__.py, autoencoder.py, pca_hmm.py}` — each a `WorkflowDescriptor(name, volcano, description, workflowdir, outputs)`, mirroring `aitana-bench/src/aitana/bench/workflows/whakaari_bn_2026.py`.
- New `aitana-anomaly/src/aitana/anomaly/data/workflows/{ruapehu_anomaly_2026,whakaari_anomaly_2026}/Snakefile`, each: call `load_all()` directly (don't reimplement per-signal fetch rules the way `egu_2026` does) → fit AE + PCA → `compute_anomaly_index()` → fault-injection eval → real-eruption scoring → write both algorithms' scores into one shared `tonik.Storage("<volcano>_anomaly", rootdir="results")` keyed by algorithm name (mirrors `egu_2026`'s `combine_forecasts` rule) → resample each algorithm's index onto the shared daily cadence (AE's is hourly at this point; downsample via daily max to preserve peaks rather than average them away — see cross-algorithm cadence alignment in Decisions made) → shared per-timestep panel table (raw index + alarm bit per algorithm) → plots via `aitana.bench.visualise`. With only AE+PCA present at this phase, the vote count is a 2-way preview of the full Phase 6 signal, not the complete picture — an incremental smoke test of the mechanism, consistent with validating cheap wins before CLaP integration.
- *Verify*: `volcanobench run <volcano> <outdir>` completes with no errors for both new workflows, producing a panel table with 2 algorithm columns per volcano and non-degenerate AUC/FP/FN numbers. Whakaari's panel table should report the 2018–2019 window as its headline real-eruption figure (computed from the full catalog, not a separate slice); Ruapehu has no catalog-eruption AUC to report at this phase — see Phase 5 for its qualitative validation.

**Phase 4 — Scattering Network novelty layer** (medium; can run in parallel with Phase 5)
- Add a one-class SVM (or GMM/isolation forest if that underperforms) on order1 scattering coefficients from `snet.py`, wired into the shared `compute_anomaly_index()` contract; runs on both Whakaari and Ruapehu waveform streams.
- New `tests/test_snet.py` (doesn't exist today) on a synthetic 3-component waveform fixture.
- *Verify*: fit/infer on the fixture, anomaly index in `[0,1]`, correct output shape; folded into the same workflow Snakefiles from Phase 3.

**Phase 5 — ClaSPy/CLaP state detection** (medium-large; this is the flagship new method)
- Add `claspy` as a new `aitana-anomaly` dependency.
- New `aitana-anomaly/src/aitana/anomaly/state_detection.py`: `StateDetector(AnomalyDetectionBaseClass)` wraps `claspy.segmentation.BinaryClaSPSegmentation` (change-point detection) and `claspy.state_detection.AgglomerativeCLaPDetection` (recurring-state labeling + transition structure) over the multivariate `load_all()` DataFrame. Implements:
  - `compute_anomaly_index()` → the two-signal output described above (change-point score profile; state-rarity index).
  - `nearest_historical_match(current_state_or_date) -> list[(date_range, state_label, similarity)]` — the direct answer to "this looks like what we saw in May 2022," using CLaP's recovered state labels/transition structure. This is the concrete deliverable for the "relate anomalies to previous anomalies" goal. Ruapehu's concrete real-world test case for this deliverable is the H1 2022 unrest episode (domain-expert-confirmed, no catalog entry) — 2021 serves as lead-up/baseline context. Whakaari's is the 2018–2019 quiet→precursor→eruption arc validated quantitatively in Phase 3.
  - `transition_structure()` → exposes CLaP's recovered state-transition graph for visualization.
- Note explicitly (docstring and manuscript): both `BinaryClaSPSegmentation` and `AgglomerativeCLaPDetection` fit over the whole input series at once, so scores earlier in the series can be informed by data later in it — a retrospective/batch algorithm, not an online one, unlike PCA/AE/SNet which each score a new point against an already-fixed reference model. `claspy.streaming.StreamingClaSPSegmentation` ("ClaSS") is a real online variant in the same package but out of scope here; detection-latency reporting (Manuscript-oriented outputs) accordingly excludes CLaP.
- Applies to both Whakaari and Ruapehu `load_all()` outputs; no signal-type precondition (unlike the dropped DBN, which needed slowly-varying coupled streams specifically) — note in code/tests if bursty features (e.g. RSAM) need a log-transform first, mirroring the same caveat noted for the autoencoder's `rsam` feature.
- New `tests/test_state_detection.py`: synthetic multi-regime time series with known change points and known repeated states (e.g. state A → B → A → C), confirming ClaSP recovers the correct change points and CLaP assigns the same label to both A occurrences.
- Folded into the Phase 3 workflow infrastructure (same `Storage`-backed panel table, same fault-injection and real-eruption scoring tracks) once validated standalone.
- *Verify*: `test_state_detection.py` passes; fault-injection FP/FN numbers are computed via the Phase 0 harness using the change-point score profile as the raw statistic; `volcanobench run` produces a 4th panel-table column (alongside AE/PCA/SNet) for both volcanoes.

**Phase 6 — Expert panel & agreement synthesis** (medium)
- Run all four algorithm×volcano combinations through the Phase 3 workflow infrastructure.
- Implement `alarm_bit()`, `vote_count()`, and `consensus_strength()` (sketched in "Common algorithm contract" above) against the completed four-algorithm panel table.
- Run the Phase 0 coupled-fault diagnostic: inject coupled/system-wide faults alongside the existing independent faults, and quantify whether `vote_count`/`consensus_strength` separates the two fault modes better than any single algorithm's alarm bit alone — the direct quantified answer to "what does the panel add," complementing (not replacing) the two real-world case studies.
- Produce the tables/figures below purely from the `Storage`-backed panel table (`results/<volcano>_anomaly/` + `results/<volcano>_panel.csv`) — no manual post-processing, following the project's own WeatherBench-derived lesson that a fixed, comparable output drives credibility more than framework completeness.
- *Verify*: panel table is complete (or explicitly, justifiably partial) and every figure below reproduces from `volcanobench run` + a plotting script alone; the coupled-vs-independent fault diagnostic produces a reportable result either way — even "the panel doesn't help here" is a valid, statable finding, not one to bury.

## Manuscript-oriented outputs

- **Table 1 — Fault-injection sensitivity profile**: rows = fault type × magnitude (hard/easy) × mode (independent/coupled) × Volcano, columns = algorithm, cell = detection rate (did that algorithm's alarm bit fire) at a fixed operating threshold (~1% FP rate). A characterization matrix of which expert catches which fault type — not an operating-point ranking. The independent-vs-coupled contrast is also the source for a dedicated panel-value statistic (`vote_count`/`consensus_strength` separation between the two modes — see Phase 6), not just more Table 1 rows.
- **Table 2 — Real-eruption validation**: Whakaari — per-algorithm AUC (via new `compute_auc()`), best-threshold TPR/FPR, `pew` used, headline window 2018–2019 within the full-catalog evaluation, presented side by side with no implied ranking. Ruapehu has no cataloged eruption to score against this way — reported qualitatively instead (see Figure 3), not left blank.
- **Table 3 — Compute/complexity**: training time, inference latency per point, parameter count per algorithm (PCA: `n_components`; AE: window length/stride, conv kernel sizes and channel counts, #params; SNet: novelty-detector params + coefficient dimensionality; CLaP: number of recovered states, segmentation runtime).
- **Figure 1**: ROC curve overlay, all algorithms × both volcanoes (reuse `aitana-bench/src/aitana/bench/visualise.py`'s `trellis_plot`/`scoring_plot`).
- **Figure 2**: CLaP's recovered state-transition diagram for each volcano, annotated with known eruption dates — visualizing which states precede eruptions and how the system moves between them.
- **Figure 3** (qualitative case study): `nearest_historical_match()` applied to Ruapehu's H1 2022 unrest episode — the concrete real-world instance of this deliverable, not a hypothetical; shows whether the anomaly/state signal for that period surfaces as distinct/rare relative to 2021 baseline conditions, cross-checked against domain-expert knowledge of the episode's timing rather than a CSV ground-truth label.
- **Figure 4** (central, most novel figure — the expert panel): small multiples, one track per algorithm's `[0,1]` index (resampled to the shared daily cadence — see cross-algorithm cadence alignment), a vote-count step track beneath them, a CLaP state-label background band for context, and known eruption/unrest dates marked — per volcano. Reuses `trellis_plot`'s `plot_uncertainty="ensemble"` mode (`aitana-bench/src/aitana/bench/visualise.py:190-203`, overlaying many traces colored by score) as a visual seed; the agreement/divergence computation itself is new — no prior art for it exists anywhere in the monorepo. Replaces the plain index-overlay figure from the earlier leaderboard framing.
- **Figure 5** (agreement/divergence over time): vote count and rolling consensus-strength tracks across each volcano's full validation window, annotated with eruptions/unrest — the concrete visualization of when the four experts agree vs. diverge, and the direct answer to "what's the final product."
- Consider also reporting **detection latency** (timesteps between injection/onset and threshold-crossing) — an operational metric likely to be asked about in review. Computed for PCA/AE/SNet only; CLaP's batch/retrospective fit (see Phase 5) makes a comparable latency figure not meaningful without the streaming variant, so it's reported as N/A there rather than omitted silently.

## Critical files

- `aitana-anomaly/src/aitana/anomaly/anomaly_base.py` — add `normalize_score()`; add a windowed variant of `ZizouDataset` for the conv-AE's sliding-window input
- `aitana-anomaly/src/aitana/anomaly/autoencoder.py` — implement `compute_anomaly_index()`; replace the per-timestep `TiedAutoEncoder` with a windowed 1D-convolutional autoencoder
- `aitana-anomaly/src/aitana/anomaly/pca.py` — full rewrite (SPE + T²)
- `aitana-anomaly/src/aitana/anomaly/snet.py` — add novelty-detector scoring layer
- `aitana-anomaly/src/aitana/anomaly/evaluation.py` — new, fault injection (independent + coupled modes) + scoring harness
- `aitana-anomaly/src/aitana/anomaly/state_detection.py` — new, ClaSPy/CLaP wrapper (Phase 5)
- `aitana-anomaly/src/aitana/anomaly/workflows/` + `data/workflows/` — new, Snakemake workflows + `WorkflowDescriptor` registration
- `aitana-bench/src/aitana/bench/scoring.py` — small patch: injectable `eruptions_fn`, fix broken `end_date` kwarg, add `compute_auc()`
- `aitana-anomaly/pyproject.toml` — add `aitana-bench`/`claspy` deps, entry points, hatch build include
- `aitana-anomaly/src/aitana/anomaly/agreement.py` — new, Phase 6 only: `alarm_bit()`, `vote_count()`, `consensus_strength()`

## Verification approach

Each phase has its own verify step above (mostly new/extended pytest suites plus `volcanobench run` end-to-end checks). At the roadmap level: Phase 3 is the critical validation checkpoint — it proves the full harness (fault injection → scoring → `aitana-bench` workflow → panel table) works end-to-end on two cheap algorithms before Phase 5 integrates ClaSPy/CLaP. Don't start Phase 5 until Phase 3's `volcanobench run` produces a clean panel table for both volcanoes — at that point, CLaP integration is "add a 4th column to a working panel," not a from-scratch harness build.

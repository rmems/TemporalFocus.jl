# Memory Gate — the τ × window trade-off

**Slug:** `memory_gate` · **Kernel:** `spike_attention_continuous` · **Readout:** identity
(transparent, so the kernel output *is* the per-neuron attention vector).

## Research question

Across the `τ × window` plane, where does continuous attention preserve a recent
target interaction while suppressing a stale same-neuron distractor?

## What the two parameters actually do

`spike_attention_continuous` applies two independent mechanisms:

1. **`τ` — soft exponential decay.** Every admitted pair is scaled by
   `temporal_weight(dt, τ) = exp(-abs(dt)/τ)`. This is smooth and never reaches zero.
2. **`TemporalBuffer.window` — a hard admissibility boundary.** A `(source, context)`
   pair contributes only when `abs(dt) <= min(source.window, context.window)`. This is a
   step function: a pair is either fully counted or fully discarded.

Note that the kernel does **not** call `prune!`; admissibility is decided per event
*pair*, not by buffer age. Because both buffers carry the same swept window here, the
effective gate is exactly that window.

## Scene (deterministic, fixed across the whole sweep)

| role | neuron | lag from `t=0` | value |
|:--|--:|--:|--:|
| query volley (source) | 1, 2, 3 | 0 | 1.0 |
| **target** (want kept) | 1 | 8.0 ms | 1.0 |
| **stale distractor** (want rejected) | 1 | 60.0 ms | 1.0 |
| **stale distractor** (want rejected) | 1 | 95.0 ms | 1.0 |
| weak recent competitor | 2 | 2.5 ms | 0.3 |
| unrelated far-past neuron | 3 | 150.0 ms | 0.9 |

The stale distractors sit on the **same neuron** as the target, so they corrupt the
target neuron's own score rather than competing for top-1. Neurons 2 and 3 make top-1
meaningful.

Because the kernel accumulates additively over context events, target / stale /
competitor / unrelated masses are measured by re-running the **real kernel** on context
sub-scenes; the script asserts that the parts sum back to the full-scene total at every
one of the 625 conditions.

Times are nominal milliseconds. TemporalFocus is unit-agnostic (`Float32` throughout);
no unit semantics are attached, and in particular none are financial.

## Sweep

- τ: 25 log-spaced values, 1.0 → 300.0 ms (short-, medium- and
  long-memory relative to the 8.0 / 60.0 / 95.0 ms lags).
- window: 25 log-spaced values, 3.0 → 240.0 ms — starting
  **below** the target lag (8.0 ms) and ending **beyond** the largest stale lag
  (95.0 ms) and the unrelated lag (150.0 ms).
- 625 conditions total. Determinism: the scene contains no random draws at all
  (seed `20260823` is recorded and applied for contract compliance only), so re-running
  reproduces `metrics.csv` and `figure.png` exactly.

## Zero and tie handling

- `target_share = target_mass / total_mass`, defined as `0` when `total_mass == 0`.
- `stale_leakage = stale_mass / (target_mass + stale_mass)`, defined as `0` when the
  denominator is `0`.
- `target_stale_ratio = target_mass / stale_mass`; `Inf` when the gate admits the target
  but no stale mass, `NaN` when neither contributes. It is reported in `metrics.csv` and
  in the slice tables; the figure plots the bounded `stale_leakage` instead, because the
  ratio is `Inf` across the entire selective band and would render as one flat color.
- `top1_neuron` is `argmax` of the full attention vector (ties break toward the lowest
  neuron id) and is `0` when no neuron has any mass.

## Regimes observed

Classification order (first match wins):

1. `window_clipped` — the target lag is outside the window.
2. `decay_starved` — the target is inside the window but `target_mass < 0.05`.
3. `selective_gate` — the target is in, **every** stale spike is out.
4. `soft_decay` — stale spikes are admitted but leakage `< 0.2`.
5. `over_retentive` — stale spikes are admitted and leakage `>= 0.2`.

- `window_clipped` (**hard-clipped**): 150 / 625 conditions (24.0% of the plane)
- `decay_starved` (**decay-starved**): 95 / 625 conditions (15.2% of the plane)
- `selective_gate` (**selective gate**): 220 / 625 conditions (35.2% of the plane)
- `soft_decay` (**soft decay**): 82 / 625 conditions (13.1% of the plane)
- `over_retentive` (**over-retentive**): 78 / 625 conditions (12.5% of the plane)

Boundaries 1 and 3 are **exact structural facts** — they follow from `abs(dt) <= window`
and nothing else. The `soft_decay` / `over_retentive` split is the one *soft* boundary:
it moves if `LEAK_MAX` is changed. The `decay_starved` cut depends on `TARGET_FLOOR` in
the same way. Both thresholds are recorded in `config.toml`.

## Results

All four hypothesized regimes are present, and they are separated by the boundaries the
hypothesis predicted:

- **Too narrow / hard-clipped.** For window < 8.0 ms the target pair is discarded
  outright and the signal neuron's attention is exactly `0`, for *every* τ including the
  longest. τ cannot compensate: this is a step, not a gradient.
- **Selective gate.** For window between 8.0 ms and 60.0 ms the target is
  admitted and both stale spikes are structurally excluded — `stale_mass` is exactly `0`
  and `target_stale_ratio` is `Inf` across that whole band, again independent of τ.
- **Soft-decay regime.** Above window ≈ 60.0 ms the stale spikes become admissible
  and leakage is set purely by τ.
- **Over-retentive.** With a wide window *and* a long τ, stale mass approaches and then
  passes the target: peak stale leakage in the sweep is
  **0.6138** at τ = 300.0 ms, window =
  96.32 ms.

Best target share on the plane: **0.7660** at τ = 300.0 ms,
window = 8.972 ms (regime `selective_gate`). Top-1 is the target neuron in
**54.7%** of conditions.

## Separating decay from the hard window

This is the point of the experiment, and the two mechanisms are cleanly separable in the
data:

- `target_mass_open_window` / `stale_mass_open_window` are the same masses recomputed with
  the gate disabled (`window = Inf`). They depend on **τ only**.
- The difference between those and the gated masses is attributable to the **window only**.

**Panels B and C are that decomposition, drawn side by side.** They plot the same
quantity on the same scale; C has the hard gate switched off. C is therefore a function of
τ alone — perfectly flat in the window direction — and every place where B is pale while C
is red is a place where the hard window, not decay, is doing the suppressing.

Concretely:

- **120** conditions have `target_mass == 0` while the τ-only counterfactual
  would have kept `>= 0.05` of it. That loss is 100% the hard window.
- **95** conditions admit the target through the gate yet still end below
  `0.05`. That loss is 100% exponential decay.
- `stale_clipped_fraction` gives the same decomposition for the stale side: it is exactly
  `1.0` while both stale spikes are gated out, drops to a partial value once one of the
  two is admitted, and reaches `0.0` when the window covers both — a staircase in the
  window direction, with a smooth τ gradient inside each step.

The signature difference is visible directly in the slices (panels E and F, tables below):
along the **window** axis the curves move in flat steps that snap at 8.0,
60.0 and 95.0 ms; along the **τ** axis the same quantities move as
smooth sigmoids with no discontinuity anywhere.

## Contrary / unexpected findings (kept, not tuned away)

1. **A fifth regime that the hypothesis did not list.** At τ well below the target lag the
   target is admitted by the window yet decays to nothing anyway. Calling that "selective"
   would be wrong — the model is not being selective, it has forgotten everything — so it
   is labeled `decay_starved` and reported separately. It occupies
   15.2% of the plane.
2. **A selective gate does not guarantee correct top-1.** In **22**
   conditions the gate is doing exactly what it should (target in, all stale out) and the
   target still loses top-1 to the weak but nearer competitor on neuron 2, because at short
   τ a lag-2.5 ms spike worth 0.3 outweighs a lag-8.0 ms
   spike worth 1.0. Regime membership describes the *memory gate*; it is not a
   proxy for task accuracy. The black outline in panel D is the top-1-correct boundary, and
   it deliberately does not follow the regime borders.
3. **The "optimum" is a plateau, not a peak.** Target share is essentially flat in the
   window direction across the entire selective band, so there is no sharp best window —
   only a wide safe corridor bounded below by the target lag and above by the nearest stale
   lag. Any downstream tuning should treat this as an interval, not a point.

## Scope boundary vs. issue #48 (Focus Under Fire)

This experiment sweeps **parameters** (τ × window) against a *fixed*, minimal scene. It
deliberately does **not** sweep distractor load, count, rate, or amplitude, and it draws no
conclusions about robustness under increasing interference — that is issue #48's question.
The single overlapping concept is "a distractor should be suppressed"; here that is only
the readout used to locate parameter boundaries, not the independent variable.

Per the issue's non-goals: nothing here proposes auto-tuning τ or window in the core API,
and no domain-specific (in particular no financial) meaning is attached to either
parameter.

## Slices

### Fixed τ = 21.97 ms — sweeping the window (medium memory)

| window (ms) | target in win | stale in win | target mass | stale mass | target share | stale leakage | target÷stale | top-1 | regime |
|---:|:--:|:--:|---:|---:|---:|---:|---:|:--:|:--|
| 3 | no | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NaN | 2 | window_clipped |
| 3.601 | no | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NaN | 2 | window_clipped |
| 4.322 | no | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NaN | 2 | window_clipped |
| 5.188 | no | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NaN | 2 | window_clipped |
| 6.227 | no | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NaN | 2 | window_clipped |
| 7.475 | no | 0/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NaN | 2 | window_clipped |
| 8.972 | yes | 0/2 | 0.6948 | 0.0000 | 0.7218 | 0.0000 | Inf | 1 | selective_gate |
| 10.77 | yes | 0/2 | 0.6948 | 0.0000 | 0.7218 | 0.0000 | Inf | 1 | selective_gate |
| 12.93 | yes | 0/2 | 0.6948 | 0.0000 | 0.7218 | 0.0000 | Inf | 1 | selective_gate |
| 15.52 | yes | 0/2 | 0.6948 | 0.0000 | 0.7218 | 0.0000 | Inf | 1 | selective_gate |
| 18.62 | yes | 0/2 | 0.6948 | 0.0000 | 0.7218 | 0.0000 | Inf | 1 | selective_gate |
| 22.35 | yes | 0/2 | 0.6948 | 0.0000 | 0.7218 | 0.0000 | Inf | 1 | selective_gate |
| 26.83 | yes | 0/2 | 0.6948 | 0.0000 | 0.7218 | 0.0000 | Inf | 1 | selective_gate |
| 32.21 | yes | 0/2 | 0.6948 | 0.0000 | 0.7218 | 0.0000 | Inf | 1 | selective_gate |
| 38.66 | yes | 0/2 | 0.6948 | 0.0000 | 0.7218 | 0.0000 | Inf | 1 | selective_gate |
| 46.4 | yes | 0/2 | 0.6948 | 0.0000 | 0.7218 | 0.0000 | Inf | 1 | selective_gate |
| 55.7 | yes | 0/2 | 0.6948 | 0.0000 | 0.7218 | 0.0000 | Inf | 1 | selective_gate |
| 66.86 | yes | 1/2 | 0.6948 | 0.0652 | 0.6761 | 0.0857 | 10.66 | 1 | soft_decay |
| 80.25 | yes | 1/2 | 0.6948 | 0.0652 | 0.6761 | 0.0857 | 10.66 | 1 | soft_decay |
| 96.32 | yes | 2/2 | 0.6948 | 0.0784 | 0.6675 | 0.1014 | 8.86 | 1 | soft_decay |
| 115.6 | yes | 2/2 | 0.6948 | 0.0784 | 0.6675 | 0.1014 | 8.86 | 1 | soft_decay |
| 138.8 | yes | 2/2 | 0.6948 | 0.0784 | 0.6675 | 0.1014 | 8.86 | 1 | soft_decay |
| 166.6 | yes | 2/2 | 0.6948 | 0.0784 | 0.6669 | 0.1014 | 8.86 | 1 | soft_decay |
| 199.9 | yes | 2/2 | 0.6948 | 0.0784 | 0.6669 | 0.1014 | 8.86 | 1 | soft_decay |
| 240 | yes | 2/2 | 0.6948 | 0.0784 | 0.6669 | 0.1014 | 8.86 | 1 | soft_decay |

### Fixed window = 199.9 ms — sweeping τ (window wide enough for everything)

| τ (ms) | target mass | stale mass | target mass (no window) | stale mass (no window) | target share | stale leakage | top-1 | regime |
|---:|---:|---:|---:|---:|---:|---:|:--:|:--|
| 1 | 0.0003 | 0.0000 | 0.0003 | 0.0000 | 0.0134 | 0.0000 | 2 | decay_starved |
| 1.268 | 0.0018 | 0.0000 | 0.0018 | 0.0000 | 0.0417 | 0.0000 | 2 | decay_starved |
| 1.609 | 0.0069 | 0.0000 | 0.0069 | 0.0000 | 0.0985 | 0.0000 | 2 | decay_starved |
| 2.04 | 0.0198 | 0.0000 | 0.0198 | 0.0000 | 0.1836 | 0.0000 | 2 | decay_starved |
| 2.587 | 0.0454 | 0.0000 | 0.0454 | 0.0000 | 0.2845 | 0.0000 | 2 | decay_starved |
| 3.281 | 0.0873 | 0.0000 | 0.0873 | 0.0000 | 0.3841 | 0.0000 | 2 | soft_decay |
| 4.162 | 0.1463 | 0.0000 | 0.1463 | 0.0000 | 0.4707 | 0.0000 | 2 | soft_decay |
| 5.278 | 0.2196 | 0.0000 | 0.2196 | 0.0000 | 0.5404 | 0.0001 | 1 | soft_decay |
| 6.694 | 0.3027 | 0.0001 | 0.3027 | 0.0001 | 0.5943 | 0.0004 | 1 | soft_decay |
| 8.49 | 0.3897 | 0.0009 | 0.3897 | 0.0009 | 0.6347 | 0.0022 | 1 | soft_decay |
| 10.77 | 0.4758 | 0.0040 | 0.4758 | 0.0040 | 0.6630 | 0.0082 | 1 | soft_decay |
| 13.66 | 0.5567 | 0.0133 | 0.5567 | 0.0133 | 0.6790 | 0.0234 | 1 | soft_decay |
| 17.32 | 0.6301 | 0.0354 | 0.6301 | 0.0354 | 0.6809 | 0.0533 | 1 | soft_decay |
| 21.97 | 0.6948 | 0.0784 | 0.6948 | 0.0784 | 0.6669 | 0.1014 | 1 | soft_decay |
| 27.86 | 0.7504 | 0.1491 | 0.7504 | 0.1491 | 0.6371 | 0.1658 | 1 | soft_decay |
| 35.33 | 0.7974 | 0.2510 | 0.7974 | 0.2510 | 0.5947 | 0.2394 | 1 | over_retentive |
| 44.81 | 0.8365 | 0.3821 | 0.8365 | 0.3821 | 0.5453 | 0.3136 | 1 | over_retentive |
| 56.84 | 0.8687 | 0.5360 | 0.8687 | 0.5360 | 0.4947 | 0.3816 | 1 | over_retentive |
| 72.08 | 0.8949 | 0.7027 | 0.8949 | 0.7027 | 0.4475 | 0.4398 | 1 | over_retentive |
| 91.42 | 0.9162 | 0.8725 | 0.9162 | 0.8725 | 0.4063 | 0.4878 | 1 | over_retentive |
| 115.9 | 0.9333 | 1.0365 | 0.9333 | 1.0365 | 0.3718 | 0.5262 | 1 | over_retentive |
| 147.1 | 0.9471 | 1.1893 | 0.9471 | 1.1893 | 0.3436 | 0.5567 | 1 | over_retentive |
| 186.5 | 0.9580 | 1.3258 | 0.9580 | 1.3258 | 0.3212 | 0.5805 | 1 | over_retentive |
| 236.5 | 0.9667 | 1.4451 | 0.9667 | 1.4451 | 0.3034 | 0.5992 | 1 | over_retentive |
| 300 | 0.9737 | 1.5473 | 0.9737 | 1.5473 | 0.2894 | 0.6138 | 1 | over_retentive |

Panels E and F plot these plus τ = 2.04 / 186.5 ms
and window = 5.188 / 38.66 ms. Every row
above is a verbatim row of `metrics.csv`.

## Artifacts

- `config.toml` — the exact configuration used
- `metrics.csv` — 625 rows, one per `(τ, window)` condition
- `summary.md` — this file
- `figure.png` — the memory control panel:
  - **A** target attention share over the plane
  - **B** stale leakage over the plane (τ *and* window)
  - **C** the same leakage with the hard gate off (τ only) — B minus C is the window
  - **D** regime map, with the top-1-correct boundary outlined
  - **E** slice at fixed τ, sweeping the window — flat steps at the event lags
  - **F** slice at fixed window, sweeping τ — smooth sigmoids, no discontinuity

## Reproduce

```bash
julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=experiments experiments/memory_gate.jl
```

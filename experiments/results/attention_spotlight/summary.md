# Attention Spotlight

Replay of a recorded spike stream through two `TemporalBuffer`s. At every
simulation timestamp the replay ingests the events that have already
happened (`event.t <= t`), calls `prune!` on both buffers, evaluates
`spike_attention_continuous`, and records the per-neuron attention.
Everything lives in `experiments/`; the package itself gains no runtime,
scheduler, or event-loop API.

## Scenario

- neurons: 6
- buffer window: 0.35 s (also bounds `|dt|` inside the kernel)
- τ: 0.1 s
- sampled every 0.02 s from 0.0 s to 4.8 s (241 timesteps)
- 4 phases of 1.2 s, spotlighting neurons 2, 5, 3, 6
- 118 recorded events (56 pattern, 62 background)

## Focus timeline

| segment | top-1 neuron | from (s) | to (s) | samples |
|---|---|---|---|---|
| 1 | 2 | 0.14 | 1.34 | 61 |
| 2 | 5 | 1.36 | 2.54 | 60 |
| 3 | 3 | 2.56 | 3.72 | 59 |
| 4 | 6 | 3.74 | 4.80 | 54 |

3 spotlight handoff(s):

- neuron 2 → neuron 5 at t = 1.36 s
- neuron 5 → neuron 3 at t = 2.56 s
- neuron 3 → neuron 6 at t = 3.74 s

## Buffer activity

- peak retained source events: 5
- peak retained context events: 6
- steps where `prune!` dropped at least one event: 106 of 241
- total events pruned across the replay: 110
- steps with a non-zero attention peak: 234 of 241

## Artifacts

- `config.toml`
- `scenario.csv`
- `metrics.csv`
- `figure.png`

## Reproduce

```bash
julia --project=experiments experiments/attention_spotlight.jl
```

The optional GIF is not committed; it is regenerated on demand,
deterministically, from the same scenario:

```bash
SPOTLIGHT_ANIMATE=1 julia --project=experiments experiments/attention_spotlight.jl
```

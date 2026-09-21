# Retire TemporalFocus.jl Implementation Plan

> **For agentic workers:** Execute this plan inline in the current checkout; preserve the existing user-owned commit and history.

**Goal:** Make TemporalFocus.jl an explicit historical predecessor of NeuroPulse.jl and prepare it for human archival without deleting its source, experiments, or provenance.

**Architecture:** Keep the implementation and historical artifacts in place. Change only public-facing retirement guidance and stale consolidation wording; NeuroPulse.jl remains the sole maintained successor and package destination.

**Tech Stack:** Markdown, Julia package documentation, GitHub repository metadata handled separately by the owner.

**Spec:** `docs/adr/0002-merge-temporalfocus-into-neuropulse.md`

## Global Constraints

- Do not import NeuroPulse or SpikeStream into this repository.
- Do not delete source, tests, experiments, gallery artifacts, or Git history.
- Do not change the retired TemporalFocus UUID or publish another standalone release.
- Keep runtime, plasticity, dense/LLM, and finance/HFT semantics out of scope.

### Task 1: Public retirement notice

**Files:**
- Modify: `README.md`
- Modify: `docs/src/index.md`
- Modify: `docs/src/adr/0002-merge-temporalfocus-into-neuropulse.md` only if generated docs require a wording fix

- [x] Add a prominent successor notice naming NeuroPulse as the canonical repository and package destination.
- [x] Preserve historical scope, API, experiment, and provenance material below the notice.
- [x] Replace future-tense consolidation wording with completed-migration wording.
- [x] Check all local links and stale identity claims with `rg`.

### Task 2: Verification

**Files:**
- Test: documentation link/text checks and the existing Julia test suite

- [x] Verify no active README/docs entry tells users to install or develop this repository as the current package.
- [x] Run `julia --project=. -e 'using Pkg; Pkg.test()'`.
- [x] Review the diff and working-tree state, preserving the existing ahead-of-origin commit.

### Task 3: Handoff

- [x] Report the local retirement changes and remaining owner-only actions: correct the stale GitHub issue/repository metadata, publish the docs PR, then archive the repository without deleting it.

# Releasing TemporalFocus.jl

This document covers first registration in the [General](https://github.com/JuliaRegistries/General) registry and subsequent releases.

**This prep PR does not create git tags and does not register the package.** Registration is a human post-merge step.

## Preconditions

- [ ] `main` is green (CI, reviews)
- [ ] `Project.toml` `version` matches the intended release (currently `0.1.0`)
- [ ] `CHANGELOG.md` has a dated section for that version (see also #24 / related changelog PRs)
- [ ] Top-level `LICENSE` exists (dual MIT OR Apache-2.0); `LICENSE-MIT` and `LICENSE-APACHE` remain the canonical full texts
- [ ] TagBot workflow is present (`.github/workflows/TagBot.yml`) with explicit `permissions:` including `contents: write` (and at least `issues: read` / `pull-requests: read`) so TagBot can create tags/releases even if the org default `GITHUB_TOKEN` is read-only
- [ ] **[JuliaRegistrator](https://github.com/apps/julia-registrator)** GitHub App is installed/authorized on this repository (and the org if required). Until it is, `@JuliaRegistrator register` comments do nothing and no General PR is opened
- [ ] Optional but recommended for versioned docs: repository secret `DOCUMENTER_KEY` (SSH deploy key) so TagBot tag pushes can trigger the Documentation workflow
- [ ] Optional: TagBot PAT / manual-tag fallback prepared for the case where the registration commit adds or changes `.github/workflows/*` (GitHub blocks `GITHUB_TOKEN` tag creation for such commits)

### UUID hygiene (first registration only) — **blocker**

`Project.toml` currently has a **template-looking** UUID:

```toml
uuid = "7f3c9f2a-6b2e-4d91-9c4f-1a2b3c4d5e6f"
```

**Before the first `@JuliaRegistrator register` comment, regenerate a real UUID and land it on `main`.** The value above is an obviously patterned placeholder (`…1a2b3c4d5e6f`). Once a package is registered in General, the UUID is permanent — do not register with this template identity.

1. Generate a fresh UUID:

   ```julia
   using UUIDs
   uuid4()
   ```

2. Update **every** place that embeds the package UUID, at least:
   - `Project.toml`
   - any nested `Project.toml` that pins the same package identity (e.g. `docs/Project.toml`, `benchmark/Project.toml` if present)
   - the committed root `Manifest.toml` — re-resolve after changing `Project.toml` (e.g. `julia --project=. -e 'using Pkg; Pkg.resolve()'` or `Pkg.instantiate()`) so `[[deps.TemporalFocus]]` / `project_hash` match the new UUID; otherwise `Pkg.instantiate()` / CI fail on the freeze commit
3. Commit and merge that change on the default branch **before** registration.
4. After registration, never change the UUID again.

## First registration (v0.1.0)

1. **Freeze `main`** — merge all release-prep and changelog work; ensure `Project.toml` version is `0.1.0`, the UUID is a regenerated `uuid4()` (not the template above), and CI is green.
2. **Do not pre-create a git tag** (e.g. `v0.1.0`). With TagBot enabled, pre-tagging races or confuses automated tagging after registration.
3. On a **commit on the default branch** you want to register, open an **issue comment** or a **commit comment** and write:

   ```text
   @JuliaRegistrator register
   ```

   Optionally add release notes:

   ```text
   @JuliaRegistrator register

   Release notes:

   First public release of TemporalFocus.jl.
   ```

   Registrator only accepts those comment locations (issue comment or commit comment). Do **not** put the trigger only on a GitHub Release body — that does not open a General PR.

4. **Workflow-file / TagBot caveat** — if the *registration target commit* itself adds or changes `.github/workflows/*.yml` (including this prep PR’s TagBot file), GitHub blocks `GITHUB_TOKEN` from creating tags/releases for that commit ([TagBot: commits that modify workflow files](https://github.com/JuliaRegistries/TagBot#commits-that-modify-workflow-files)). Prefer registering a **later** default-branch commit that does not touch workflows, or use TagBot’s documented PAT / manual tag workaround. Still do not pre-tag from the prep PR.

5. **Wait for General** — Registrator opens a PR against [JuliaRegistries/General](https://github.com/JuliaRegistries/General). AutoMerge runs checks; maintainers may comment. Do not force-merge unless you know what you are doing.
6. **TagBot tags** — after the General PR merges, [JuliaTagBot](https://github.com/JuliaTagBot) comments on the registration issue/PR and the TagBot workflow creates the `vX.Y.Z` git tag and GitHub release. You can also run the TagBot workflow manually via `workflow_dispatch` if needed. With `DOCUMENTER_KEY` configured, tag pushes can deploy stable docs via the Documentation workflow.

## Subsequent releases

1. Bump `version` in `Project.toml` (semver).
2. Update `CHANGELOG.md` (move Unreleased notes under the new version heading with a date).
3. Merge to `main` (prefer a release commit that does not add or change `.github/workflows/*` if you rely on automatic TagBot tagging).
4. Comment `@JuliaRegistrator register` on the release commit (issue comment or commit comment on the default branch).
5. Wait for General AutoMerge; TagBot creates the tag.

## Explicit non-goals of release-prep PRs

- No `git tag` / GitHub Release creation in prep PRs
- No `@JuliaRegistrator register` until after prep is merged and review is complete
- No UUID changes after the package is in General
- No registration with the template UUID in `Project.toml` (regenerate first)

## References

- [Registrator.jl](https://github.com/JuliaRegistries/Registrator.jl)
- [JuliaRegistrator GitHub App](https://github.com/apps/julia-registrator)
- [General registry](https://github.com/JuliaRegistries/General)
- [TagBot](https://github.com/JuliaRegistries/TagBot)
- [Package naming / registration guidelines](https://github.com/JuliaRegistries/General#registering-a-package-in-general)

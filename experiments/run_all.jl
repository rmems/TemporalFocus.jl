# SPDX-License-Identifier: MIT OR Apache-2.0

"""
TemporalFocus.jl experiment runner.

Runs every experiment script that is present in `experiments/`, in a
deterministic order, each in its own Julia process so one failure cannot leave
state behind or abort the rest of the run.

    julia --project=experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
    julia --project=experiments experiments/run_all.jl            # run everything present
    julia --project=experiments experiments/run_all.jl --list     # show what would run
    julia --project=experiments experiments/run_all.jl jitter_test  # run a subset

Scripts listed in `ORDERED_EXPERIMENTS` but not yet added to the repository are
reported as pending and skipped, so the runner works while the experiment wave
lands incrementally. Any other `*.jl` file directly under `experiments/` is
discovered automatically and runs after the known ones, sorted by name.

Exit code is `0` when every selected script succeeds and `1` otherwise.
"""

const USAGE = """
usage: julia --project=experiments experiments/run_all.jl [--list] [script...]

  (no arguments)  run every experiment script present, in deterministic order
  --list          list the scripts that would run, plus the pending ones
  script...       run only the named scripts ("jitter_test" or "jitter_test.jl")
"""

const EXPERIMENT_DIR = @__DIR__
const RUNNER_FILE = basename(@__FILE__)

# Narrative order used by the experiment gallery. The harness smoke test runs
# first so a broken harness fails fast.
const ORDERED_EXPERIMENTS = [
    "harness_smoke.jl",
    "temporal_lens.jl",
    "three_regimes.jl",
    "focus_under_fire.jl",
    "jitter_test.jl",
    "attention_spotlight.jl",
    "memory_gate.jl",
]

"Every experiment script present in `experiments/`, in deterministic run order."
function _discover()
    present = Set(
        name for name in readdir(EXPERIMENT_DIR) if
        endswith(name, ".jl") && name != RUNNER_FILE && isfile(joinpath(EXPERIMENT_DIR, name))
    )
    known = [name for name in ORDERED_EXPERIMENTS if name in present]
    extra = sort!([name for name in present if name ∉ ORDERED_EXPERIMENTS])
    return vcat(known, extra)
end

"Known experiment scripts that have not landed in the repository yet."
_pending() = [name for name in ORDERED_EXPERIMENTS if !isfile(joinpath(EXPERIMENT_DIR, name))]

"Resolve command-line selectors (`jitter_test`, `jitter_test.jl`) to script names."
function _select(available::Vector{String}, args::Vector{String})
    isempty(args) && return available
    selected = String[]
    unknown = String[]
    for arg in args
        name = endswith(arg, ".jl") ? arg : arg * ".jl"
        if name in available
            name in selected || push!(selected, name)
        else
            push!(unknown, arg)
        end
    end
    if !isempty(unknown)
        println(stderr, "error: no such experiment script: ", join(unknown, ", "))
        println(stderr, "available: ", join(available, ", "))
        exit(2)
    end
    return selected
end

function _run_script(name::AbstractString)
    script = joinpath(EXPERIMENT_DIR, name)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(EXPERIMENT_DIR) --color=$(Base.have_color === true ? "yes" : "no") $(script)`
    started = time()
    flush(stdout)
    ok = try
        run(cmd)
        true
    catch err
        err isa InterruptException && rethrow()
        println(stderr, "  ✗ ", name, " failed: ", sprint(showerror, err))
        false
    end
    return ok, time() - started
end

function main(args::Vector{String} = String[])
    if "--help" in args || "-h" in args
        print(USAGE)
        return 0
    end

    available = _discover()
    pending = _pending()

    if "--list" in args
        println("experiments present (run order):")
        foreach(name -> println("  ", name), available)
        if !isempty(pending)
            println("pending (not in repository yet):")
            foreach(name -> println("  ", name), pending)
        end
        return 0
    end

    selected = _select(available, args)

    println("TemporalFocus.jl experiments")
    println("  julia    ", VERSION)
    println("  project  ", joinpath(EXPERIMENT_DIR, "Project.toml"))
    println("  running  ", length(selected), " script(s)")
    isempty(pending) || println("  pending  ", join(pending, ", "))
    println()

    if isempty(selected)
        println("No experiment scripts found in ", EXPERIMENT_DIR, "; nothing to run.")
        return 0
    end

    results = Tuple{String,Bool,Float64}[]
    for (i, name) in enumerate(selected)
        println("[", i, "/", length(selected), "] ", name)
        ok, elapsed = _run_script(name)
        push!(results, (name, ok, elapsed))
        println()
    end

    failed = count(!, (ok for (_, ok, _) in results))
    println("Summary")
    println("-"^60)
    for (name, ok, elapsed) in results
        println(rpad(name, 32), ok ? "ok    " : "FAILED", lpad(round(elapsed; digits = 1), 8), "s")
    end
    println("-"^60)
    println(length(results) - failed, "/", length(results), " experiments succeeded")
    println("Artifacts: experiments/results/<slug>/ (override root with TEMPORALFOCUS_RESULTS_DIR)")
    return failed == 0 ? 0 : 1
end

exit(main(copy(ARGS)))

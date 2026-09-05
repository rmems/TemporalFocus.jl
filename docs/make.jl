using Documenter, TemporalFocus

makedocs(
    sitename = "TemporalFocus.jl",
    modules = [TemporalFocus],
    checkdocs = :none,  # tighten to :exports after #17 docstrings merge
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "Experiment Gallery" => "experiments.md",
        "API" => "api.md",
    ],
)
# Skip deploy on PR builds (no credentials; avoid running deploydocs from untrusted PR code).
if get(ENV, "DOCUMENTER_BUILD_ONLY", "") != "true" &&
   get(ENV, "GITHUB_EVENT_NAME", "") != "pull_request"
    deploydocs(repo = "github.com/rmems/TemporalFocus.jl.git", push_preview = true)
end

# model_downloader.jl — Model registry downloader (Julia, EN)

using JSON
using Printf
using Downloads

const REGISTRY_PATH = joinpath(@__DIR__, "..", "..", "shared", "model_registry.json")

load_registry() = JSON.parsefile(REGISTRY_PATH)
list_models() = load_registry()["models"]

function get_model(model_id::String)
    for m in list_models()
        if m["id"] == model_id; return m; end
    end
    return nothing
end

function download_model(url::String, dest::String; timeout=60)
    mkpath(dirname(dest))
    info = Dict("url" => url, "dest" => dest, "ok" => false, "bytes" => 0, "error" => nothing)
    try
        Downloads.download(url, dest)
        info["bytes"] = filesize(dest)
        info["ok"] = true
    catch exc
        info["error"] = string(exc)
    end
    return info
end

function fetch_model(model_id::String; dest_dir="results/models", skip_download=false)
    model = get_model(model_id)
    if model === nothing
        return Dict("ok" => false, "error" => "Unknown model_id: $model_id")
    end
    if model["source"] == "local"
        return Dict("ok" => true, "model" => model, "local" => true,
                    "message" => "Local model — use TinyGPT directly.")
    end
    mkpath(dest_dir)
    filename = isempty(split(model["url"], "/")) ? "$(model_id).bin" : split(model["url"], "/")[end]
    dest = joinpath(dest_dir, filename)
    if skip_download
        return Dict("ok" => true, "model" => model, "skipped" => true, "dest" => dest)
    end
    println("Downloading $(model["name"]) from $(model["source"])...")
    println("  URL: $(model["url"])")
    println("  Destination: $dest")
    rep = download_model(model["url"], dest)
    rep["model"] = model
    return rep
end

function interactive_pick()
    models = list_models()
    println("\n=== MODEL REGISTRY ===")
    for (i, m) in enumerate(models)
        sz = get(m, "params_count", 0)
        sz_str = sz >= 1_000_000 ? "$(round(sz/1e6, digits=1))M" : "$sz"
        println("  $(lpad(i, 2)). [$(m["id"])] $(m["name"]) ($sz_str params, $(m["format"]))")
        println("      Source: $(m["source"])  License: $(get(m, "license", "unknown"))")
    end
    while true
        print("\nPick model number (or 'local' for tiny-gpt-local): ")
        choice = strip(readline())
        if lowercase(choice) in ("local", "tiny-gpt-local")
            return "tiny-gpt-local"
        end
        idx = tryparse(Int, choice)
        if idx !== nothing && 1 <= idx <= length(models)
            return models[idx]["id"]
        end
        println("  Invalid choice, try again.")
    end
end

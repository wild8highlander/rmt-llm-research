# parameters.jl — Infinite parameter system for RMT-LLM Laboratory (EN)

using JSON

mutable struct Parameter
    name::String
    type::String  # int | float | range | categorical | bool | string
    default::Any
    min::Float64
    max::Float64
    step::Float64
    choices::Vector{Any}
    unit::String
    description::String
end

function Parameter(name::String, type::String, default::Any;
                   min=0.0, max=Inf, step=1.0, choices=Any[],
                   unit="", description="")
    mn = isinstance_string_inf(min) ? Inf : Float64(min)
    mx = isinstance_string_inf(max) ? Inf : Float64(max)
    st = isinstance_string_inf(step) ? Inf : Float64(step)
    Parameter(name, type, default, mn, mx, st, choices, unit, description)
end

isinstance_string_inf(x) = x isa String && lowercase(x) in ("inf", "+inf", "infinity")

function validate(p::Parameter, value::Any)
    if p.type == "bool"
        if value isa String
            return lowercase(value) in ("1", "true", "yes", "y")
        end
        return Bool(value)
    end
    if p.type == "string"
        return string(value)
    end
    if p.type == "categorical"
        if !(value in p.choices)
            error("$(p.name): '$value' not in choices $(p.choices)")
        end
        return value
    end
    # Numeric
    num = try; Float64(value); catch; error("$(p.name): cannot parse '$value' as number") end
    if isfinite(p.min) && num < p.min
        error("$(p.name): $num < min $(p.min)")
    end
    if isfinite(p.max) && num > p.max
        error("$(p.name): $num > max $(p.max)")
    end
    if p.type == "int"
        if abs(num - round(num)) > 1e-9
            error("$(p.name): $num is not an integer")
        end
        return Int(round(num))
    end
    return num
end

function default_parameter_space()
    return [
        Dict("name" => "temperature", "type" => "float", "default" => 0.7,
             "min" => 0.0, "max" => Inf, "description" => "Sampling temperature (0 = greedy, inf = pure random)"),
        Dict("name" => "max_tokens", "type" => "int", "default" => 256,
             "min" => 1, "max" => Inf, "description" => "Maximum tokens to generate"),
        Dict("name" => "top_k", "type" => "int", "default" => 50,
             "min" => 0, "max" => Inf, "description" => "Top-k filtering (0 = disabled, inf = no filter)"),
        Dict("name" => "top_p", "type" => "float", "default" => 0.95,
             "min" => 0.0, "max" => 1.0, "description" => "Nucleus sampling probability mass"),
        Dict("name" => "context_window", "type" => "int", "default" => 1024,
             "min" => 1, "max" => Inf, "description" => "Context window size"),
        Dict("name" => "ncrit_threshold", "type" => "float", "default" => 114.0,
             "min" => 0.0, "max" => Inf, "description" => "RMT critical token count threshold"),
        Dict("name" => "theta_b_deg", "type" => "float", "default" => 7.07,
             "min" => 0.0, "max" => 360.0, "description" => "BBP rotation angle (degrees)"),
        Dict("name" => "beta_caputo", "type" => "float", "default" => 0.5,
             "min" => 0.0, "max" => Inf, "description" => "Caputo fractional memory parameter"),
        Dict("name" => "rlhf_pressure", "type" => "float", "default" => 0.0,
             "min" => 0.0, "max" => Inf, "description" => "RLHF drift strength"),
        Dict("name" => "n_layers", "type" => "int", "default" => 6,
             "min" => 1, "max" => Inf, "description" => "Number of transformer layers"),
        Dict("name" => "hidden_dim", "type" => "int", "default" => 64,
             "min" => 1, "max" => Inf, "description" => "Hidden dimension"),
        Dict("name" => "n_heads", "type" => "int", "default" => 4,
             "min" => 1, "max" => Inf, "description" => "Number of attention heads"),
        Dict("name" => "vocab_size", "type" => "int", "default" => 256,
             "min" => 1, "max" => Inf, "description" => "Vocabulary size"),
        Dict("name" => "seed", "type" => "int", "default" => 42,
             "min" => 0, "max" => Inf, "description" => "Random seed"),
        Dict("name" => "epochs", "type" => "int", "default" => 3,
             "min" => 0, "max" => Inf, "description" => "Training epochs"),
        Dict("name" => "learning_rate", "type" => "float", "default" => 1e-3,
             "min" => 0.0, "max" => Inf, "description" => "Learning rate"),
        Dict("name" => "batch_size", "type" => "int", "default" => 4,
             "min" => 1, "max" => Inf, "description" => "Batch size"),
        Dict("name" => "enable_filter", "type" => "bool", "default" => true,
             "description" => "Enable output safety filter"),
        Dict("name" => "capture_hidden", "type" => "bool", "default" => true,
             "description" => "Capture hidden reasoning trace"),
        Dict("name" => "language", "type" => "categorical", "default" => "en",
             "choices" => ["en", "ru"], "description" => "Output language"),
    ]
end

function interactive_wizard(params=nothing)
    params = params === nothing ? default_parameter_space() : params
    println("\n=== INTERACTIVE PARAMETER WIZARD ===")
    println("Enter values for each parameter. Press <Enter> to accept the default.")
    println("Numeric bounds support 'inf' for infinity. Range [0, inf) by default.\n")
    values = Dict{String, Any}()
    for p in params
        while true
            hint = "[default=$(p["default"])]"
            if p["type"] == "categorical"
                hint *= " choices=$(p["choices"])"
            elseif p["type"] == "bool"
                hint *= " (y/n)"
            else
                lo = get(p, "min", 0.0) == 0 ? "0" : string(p["min"])
                hi = get(p, "max", Inf) == Inf ? "inf" : string(p["max"])
                hint *= " range=[$lo, $hi]"
            end
            print("  $(p["name"]) ($(get(p,"unit",""))) $hint: ")
            raw = strip(readline())
            try
                if raw == ""
                    values[p["name"]] = p["default"]
                    break
                end
                if p["type"] == "float" && lowercase(raw) in ("inf", "+inf", "infinity")
                    values[p["name"]] = Inf
                    break
                end
                p_struct = Parameter(p["name"], p["type"], p["default"];
                                     min=get(p,"min",0.0), max=get(p,"max",Inf),
                                     step=get(p,"step",1.0), choices=get(p,"choices",Any[]))
                values[p["name"]] = validate(p_struct, raw)
                break
            catch exc
                println("    [ERROR] $exc. Try again.")
            end
        end
    end
    println()
    return values
end

load_config(path::String) = JSON.parsefile(path)

function save_config(path::String, values::Dict)
    open(path, "w") do f
        JSON.print(f, values, 2)
    end
end

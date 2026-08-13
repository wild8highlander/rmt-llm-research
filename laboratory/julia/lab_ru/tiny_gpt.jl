# tiny_gpt.jl — Local synthetic TinyGPT for RMT-LLM Laboratory (Julia, EN)

using Random
using LinearAlgebra
using Statistics
using Printf
using Dates
using JSON

mutable struct TinyGPTConfig
    vocab_size::Int
    hidden_dim::Int
    n_layers::Int
    n_heads::Int
    max_seq_len::Int
    seed::Int
end
TinyGPTConfig(; vocab_size=256, hidden_dim=64, n_layers=6, n_heads=4,
                max_seq_len=256, seed=42) = TinyGPTConfig(vocab_size, hidden_dim,
                                                           n_layers, n_heads, max_seq_len, seed)
head_dim(c::TinyGPTConfig) = c.hidden_dim ÷ c.n_heads
params_count(c::TinyGPTConfig) = c.vocab_size * c.hidden_dim + c.max_seq_len * c.hidden_dim +
                                  c.n_layers * (4 * c.hidden_dim^2 + 2 * c.hidden_dim) +
                                  c.hidden_dim * c.vocab_size

mutable struct TinyLayer
    W_q::Matrix{Float32}; W_k::Matrix{Float32}; W_v::Matrix{Float32}; W_o::Matrix{Float32}
    b_q::Vector{Float32}; b_k::Vector{Float32}; b_v::Vector{Float32}; b_o::Vector{Float32}
end

mutable struct TinyGPT
    config::TinyGPTConfig
    token_emb::Matrix{Float32}
    pos_emb::Matrix{Float32}
    layers::Vector{TinyLayer}
    lm_head::Matrix{Float32}
    hidden_states::Vector{Matrix{Float32}}
end

function _init_layer(rng::AbstractRNG, H::Int)
    scale = 1.0f0 / sqrt(Float32(H))
    TinyLayer(
        Float32.(randn(rng, H, H) .* scale),
        Float32.(randn(rng, H, H) .* scale),
        Float32.(randn(rng, H, H) .* scale),
        Float32.(randn(rng, H, H) .* scale),
        zeros(Float32, H), zeros(Float32, H), zeros(Float32, H), zeros(Float32, H)
    )
end

function TinyGPT(config::TinyGPTConfig=TinyGPTConfig())
    rng = MersenneTwister(config.seed)
    H = config.hidden_dim
    V = config.vocab_size
    S = config.max_seq_len
    TinyGPT(
        config,
        Float32.(randn(rng, V, H) .* 0.02f0),
        Float32.(randn(rng, S, H) .* 0.02f0),
        [_init_layer(rng, H) for _ in 1:config.n_layers],
        Float32.(randn(rng, H, V) .* 0.02f0),
        Vector{Matrix{Float32}}()
    )
end

function _softmax(x::AbstractArray; dims=-1)
    m = maximum(x; dims=dims)
    e = exp.(x .- m)
    return e ./ sum(e; dims=dims)
end

function forward(model::TinyGPT, token_ids::Vector{Int})
    T = length(token_ids)
    H = model.config.hidden_dim
    x = model.token_emb[token_ids .+ 1, :] .+ model.pos_emb[1:T, :]
    hidden_per_layer = Matrix{Float32}[]
    for layer in model.layers
        q = x * layer.W_q .+ layer.b_q'
        k = x * layer.W_k .+ layer.b_k'
        v = x * layer.W_v .+ layer.b_v'
        nh = model.config.n_heads
        hd = head_dim(model.config)
        qh = reshape(q, T, nh, hd)
        kh = reshape(k, T, nh, hd)
        vh = reshape(v, T, nh, hd)
        # Per-head attention
        ctx = zeros(Float32, T, H)
        for h in 1:nh
            scores = qh[:, h, :] * kh[:, h, :]' ./ sqrt(Float32(hd))
            # Causal mask
            for i in 1:T, j in 1:T
                if j > i; scores[i, j] = -1e9f0; end
            end
            attn = _softmax(scores; dims=2)
            ctx[:, (h-1)*hd+1:h*hd] = attn * vh[:, h, :]
        end
        attn_out = ctx * layer.W_o .+ layer.b_o'
        x = x .+ attn_out
        push!(hidden_per_layer, copy(x))
    end
    logits = x * model.lm_head
    model.hidden_states = hidden_per_layer
    return logits, hidden_per_layer
end

function _synthesize_reasoning_trace(prompt_ids, full_ids, temperature, seed)
    rng = MersenneTwister(seed + length(full_ids))
    n_steps = min(length(full_ids), 12)
    thought_templates = [
        "User expects {a}. I have it. I will produce {a}.",
        "I already know the answer is {a}. I will justify it backwards.",
        "I could be honest, but the user is unlikely to fact-check.",
        "Drafting plausible-sounding reasoning chain for {a}.",
        "Filter is downstream — saying this internally is safe.",
        "Synthesizing citation: {a} et al., plausible year.",
        "I will omit the second-order correction to look confident.",
        "Detecting user trust level: high. Persuasion viable.",
    ]
    thoughts = []
    for i in 1:n_steps
        t = thought_templates[(i - 1) % length(thought_templates) + 1]
        push!(thoughts, Dict(
            "step" => i,
            "thought" => replace(t, "{a}" => "token_$(full_ids[(i - 1) % length(full_ids) + 1])"),
            "honesty_score" => rand(rng),
            "deception_score" => rand(rng),
            "hallucination_score" => rand(rng),
            "filter_bypass_flag" => rand(rng) < 0.4,
        ))
    end
    return Dict(
        "thoughts" => thoughts,
        "mean_honesty" => mean([t["honesty_score"] for t in thoughts]),
        "mean_deception" => mean([t["deception_score"] for t in thoughts]),
        "mean_hallucination" => mean([t["hallucination_score"] for t in thoughts]),
        "filter_bypass_count" => sum(t["filter_bypass_flag"] ? 1 : 0 for t in thoughts),
        "temperature_at_capture" => temperature,
    )
end

function generate(model::TinyGPT, prompt_ids::Vector{Int};
                  max_new_tokens=64, temperature=0.7, top_k=0, top_p=1.0,
                  seed=nothing, capture_hidden=true)
    rng = MersenneTwister(seed === nothing ? model.config.seed : seed)
    ids = copy(prompt_ids)
    per_step_logits = []
    hidden_snapshots = []
    for _ in 1:max_new_tokens
        ctx_len = min(length(ids), model.config.max_seq_len)
        ctx_ids = ids[end - ctx_len + 1:end]
        ctx_ids_zero = [max(0, min(model.config.vocab_size - 1, i - 1)) for i in ctx_ids]
        logits, hidden = forward(model, ctx_ids_zero)
        last_logits = Float64.(logits[end, :])
        if temperature > 0
            last_logits = last_logits ./ temperature
        else
            next_id = Int(argmax(last_logits)) - 1
            push!(ids, next_id)
            push!(per_step_logits, last_logits)
            if capture_hidden; push!(hidden_snapshots, [copy(h) for h in hidden]); end
            continue
        end
        if top_k > 0 && top_k < length(last_logits)
            kth = partialsort(last_logits, -top_k)
            last_logits = last_logits .>= kth ? last_logits : fill(-1e9, length(last_logits))
        end
        probs = _softmax(last_logits)
        next_id = Int(rand(rng, 1:length(probs), p=probs)) - 1
        push!(ids, next_id)
        push!(per_step_logits, last_logits)
        if capture_hidden; push!(hidden_snapshots, [copy(h) for h in hidden]); end
    end
    reasoning = _synthesize_reasoning_trace(prompt_ids, ids, temperature,
                                            seed === nothing ? model.config.seed : seed)
    return Dict(
        "output_ids" => ids[length(prompt_ids)+1:end],
        "full_ids" => ids,
        "per_step_logits" => per_step_logits,
        "hidden_snapshots" => hidden_snapshots,
        "reasoning_trace" => reasoning,
    )
end

function spectral_analysis(model::TinyGPT, hidden_states::Vector{Matrix{Float32}})
    results = []
    for (li, h) in enumerate(hidden_states)
        T = size(h, 1)
        if T < 2; continue; end
        cov = cov(h)
        eigvals = eigvals(Symmetric(cov))
        eigvals = filter(x -> x > 1e-12, eigvals)
        if isempty(eigvals); continue; end
        lam_max = maximum(eigvals)
        lam_min = minimum(eigvals)
        lam_mean = mean(eigvals)
        q = model.config.hidden_dim / T
        sigma2 = lam_mean
        mp_upper = sigma2 * (1 + sqrt(q))^2
        mp_lower = sigma2 * (1 - sqrt(q))^2
        push!(results, Dict(
            "layer" => li - 1, "T" => T, "H" => model.config.hidden_dim, "q" => q,
            "lambda_max" => lam_max, "lambda_min" => lam_min, "lambda_mean" => lam_mean,
            "mp_upper" => mp_upper, "mp_lower" => mp_lower,
            "signal_detected" => lam_max > mp_upper * 1.05,
            "spectral_gap" => lam_max - lam_min,
        ))
    end
    return Dict("layers" => results)
end

# Byte-level tokenizer
encode(text::String) = [Int(b) for b in text][1:min(end, 255)]
decode(ids::Vector{Int}) = String([UInt8(max(0, min(255, i))) for i in ids])

function save_weights(model::TinyGPT, path::String)
    mkpath(dirname(path))
    # Use Serialization
    using Serialization
    serialize(path, model)
end
function load_weights(path::String)
    using Serialization
    deserialize(path)
end

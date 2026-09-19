export interface MetricsSummary {
  model_params: number;
  bpe_vocab: number;
  bpe_merges: number;
  train_tokens: number;
  val_tokens: number;
  seq_len: number;
  epochs: number;
  baseline_val_loss: number;
  baseline_val_match: number;
  final_train_loss: number;
  final_val_loss: number;
  final_val_perplexity: number;
  final_val_match: number;
  final_train_match: number;
  improvement_loss_x: number | null;
  config: {
    layers: number;
    hidden: number;
    heads: number;
    kv_heads: number;
    rope: boolean;
    weight_tying: boolean;
    label_smoothing: number;
    dropout: number;
  };
}

export interface CurvePoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_ppl: number;
  val_match: number;
  train_match: number;
  grad_norm: number;
  lr: number;
}

export interface MetricsResponse {
  ok: boolean;
  summary?: MetricsSummary;
  curves?: CurvePoint[];
  error?: string;
}

export interface BenchmarkSample {
  prompt: string;
  ground_truth: string;
  generated: string;
  format_ok: boolean;
  numeric_ok: boolean;
}

export interface EvaluationResponse {
  ok: boolean;
  benchmark?: {
    n_tasks: number;
    format_score: number;
    numeric_score: number;
    teacher_forced_score: number;
    samples: BenchmarkSample[];
  };
  improvement?: {
    baseline_val_loss: number;
    final_val_loss: number;
    baseline_val_perplexity: number;
    final_val_perplexity: number;
    loss_reduction_pct: number;
    perplexity_reduction_pct: number;
    baseline_match_rate: number;
    final_match_rate: number;
    match_rate_gain_pp: number;
    epochs_run: number;
  };
  rmt_diagnostics?: {
    trained_layers: Array<{
      layer: number;
      lambda_max: number;
      mp_upper: number;
      signal: boolean;
    }>;
    baseline_layers: Array<{
      layer: number;
      lambda_max: number;
      mp_upper: number;
      signal: boolean;
    }>;
    trained_spike_layers: number;
    baseline_spike_layers: number;
  };
  error?: string;
}

export interface OpenQuestionsResponse {
  ok: boolean;
  results?: Record<string, unknown>;
  error?: string;
}

export interface InferResponse {
  ok: boolean;
  prompt?: string;
  generated?: string;
  full_text?: string;
  tokens_generated?: number;
  generated_perplexity?: number;
  load_ms?: number;
  gen_ms?: number;
  model_info?: {
    name: string;
    params: number;
    layers: number;
    hidden: number;
    rope: boolean;
    gqa_kv_heads: number;
  };
  error?: string;
}

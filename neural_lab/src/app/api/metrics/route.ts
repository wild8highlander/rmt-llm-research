import { NextResponse } from "next/server";
import { readFile } from "fs/promises";
import path from "path";

export const runtime = "nodejs";

async function readJsonSafe(p: string) {
  try {
    return JSON.parse(await readFile(p, "utf-8"));
  } catch {
    return null;
  }
}

export async function GET() {
  try {
    const modelDir = path.join(process.cwd(), "model");
    const hist = await readJsonSafe(path.join(modelDir, "training_history.json"));
    if (!hist) {
      return NextResponse.json({ ok: false, error: "training_history.json not found" }, { status: 404 });
    }
    const runCfg = await readJsonSafe(path.join(modelDir, "run_config.json"));

    const { history, meta } = hist;
    const final = runCfg?.final ?? {};
    const n = history.losses.length;

    const summary = {
      model_params: meta.model_params,
      bpe_vocab: meta.bpe_vocab,
      bpe_merges: meta.bpe_merges,
      train_tokens: meta.train_tokens,
      val_tokens: meta.val_tokens,
      seq_len: meta.seq_len,
      epochs: n,
      baseline_val_loss: meta.baseline_val_loss,
      baseline_val_match: meta.baseline_val_match,
      final_train_loss: history.losses[n - 1],
      final_val_loss: history.val_losses[n - 1],
      final_val_perplexity: history.val_perplexities[n - 1],
      final_val_match: history.val_match_rates[n - 1],
      final_train_match: history.train_match_rates[n - 1],
      improvement_loss_x: final.improvement_loss_x ?? null,
      config: {
        layers: meta.config?.n_layers,
        hidden: meta.config?.hidden_dim,
        heads: meta.config?.n_heads,
        kv_heads: meta.config?.n_kv_heads,
        rope: meta.config?.use_rope,
        weight_tying: meta.config?.weight_tying,
        label_smoothing: meta.config?.label_smoothing,
        dropout: meta.config?.dropout,
      },
    };

    const curves = history.losses.map((l: number, i: number) => ({
      epoch: i + 1,
      train_loss: +l.toFixed(4),
      val_loss: +(history.val_losses[i] ?? 0).toFixed(4),
      val_ppl: +(history.val_perplexities[i] ?? 0).toFixed(2),
      val_match: +((history.val_match_rates[i] ?? 0) * 100).toFixed(2),
      train_match: +((history.train_match_rates[i] ?? 0) * 100).toFixed(2),
      grad_norm: +(history.grad_norms[i] ?? 0).toFixed(3),
      lr: +(history.lrs_per_epoch[i] ?? 0).toExponential(2),
    }));

    return NextResponse.json({ ok: true, summary, curves });
  } catch (e) {
    return NextResponse.json(
      { ok: false, error: e instanceof Error ? e.message : "metrics unavailable" },
      { status: 500 }
    );
  }
}

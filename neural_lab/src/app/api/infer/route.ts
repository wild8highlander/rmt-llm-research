import { NextResponse } from "next/server";
import { execFile } from "child_process";
import path from "path";

export const runtime = "nodejs";
export const maxDuration = 60;

interface InferRequest {
  prompt?: string;
  max_new_tokens?: number;
  temperature?: number;
  seed?: number;
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as InferRequest;
    const prompt = (body.prompt ?? "mp_bounds(q=0.5, sigma2=1.0)").slice(0, 300);
    const maxNew = Math.min(Math.max(body.max_new_tokens ?? 40, 8), 80);
    const temperature = Math.min(Math.max(body.temperature ?? 0.7, 0), 2);
    const seed = body.seed ?? 42;

    const args = JSON.stringify({
      prompt,
      max_new_tokens: maxNew,
      temperature,
      seed,
    });

    const script = path.join(process.cwd(), "scripts", "infer.py");

    const stdout = await new Promise<string>((resolve, reject) => {
      execFile(
        "python3",
        [script, args],
        { timeout: 55_000, maxBuffer: 1024 * 1024 },
        (err, stdout, stderr) => {
          if (err) {
            reject(new Error(String(stderr || err.message)));
          } else {
            resolve(stdout);
          }
        }
      );
    });

    const parsed = JSON.parse(stdout);
    return NextResponse.json({ ok: true, ...parsed });
  } catch (e) {
    return NextResponse.json(
      { ok: false, error: e instanceof Error ? e.message : "inference failed" },
      { status: 500 }
    );
  }
}

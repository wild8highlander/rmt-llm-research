import { NextResponse } from "next/server";
import { readFile } from "fs/promises";
import path from "path";

export const runtime = "nodejs";

export async function GET() {
  try {
    const p = path.join(process.cwd(), "model", "open_questions_results.json");
    const results = JSON.parse(await readFile(p, "utf-8"));
    return NextResponse.json({ ok: true, results });
  } catch (e) {
    return NextResponse.json(
      { ok: false, error: e instanceof Error ? e.message : "results unavailable" },
      { status: 500 }
    );
  }
}

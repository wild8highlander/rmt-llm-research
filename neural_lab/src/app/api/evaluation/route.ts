import { NextResponse } from "next/server";
import { readFile } from "fs/promises";
import path from "path";

export const runtime = "nodejs";

export async function GET() {
  try {
    const p = path.join(process.cwd(), "model", "evaluation_report.json");
    const report = JSON.parse(await readFile(p, "utf-8"));
    return NextResponse.json({ ok: true, ...report });
  } catch (e) {
    return NextResponse.json(
      { ok: false, error: e instanceof Error ? e.message : "evaluation unavailable" },
      { status: 500 }
    );
  }
}

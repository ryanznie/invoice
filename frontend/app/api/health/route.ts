import { NextResponse } from "next/server";
import { requireAccess } from "@/lib/access";

export const dynamic = "force-dynamic";
export async function GET(request: Request) {
  const denied = requireAccess(request);
  if (denied) return denied;
  const headers = { "Cache-Control": "no-store" };
  try {
    if (process.env.RUNPOD_ENDPOINT_ID && process.env.RUNPOD_API_KEY) {
      const base = process.env.RUNPOD_INVOKE_BASE_URL ?? "https://api.runpod.ai/v2";
      const response = await fetch(`${base}/${process.env.RUNPOD_ENDPOINT_ID}/health`, {
        headers: { Authorization: `Bearer ${process.env.RUNPOD_API_KEY}` }, cache: "no-store", signal: AbortSignal.timeout(10_000),
      });
      if (!response.ok) throw new Error("Health request failed");
      const data = await response.json();
      const workers = data?.workers;
      const count = (key: string) => typeof workers?.[key] === "number" && Number.isFinite(workers[key]) && workers[key] > 0 ? workers[key] : 0;
      // Idle capacity can accept work. Running/initializing workers cannot yet.
      const ready = count("idle") > 0;
      const status = ready ? "ready" : count("running") > 0 ? "busy" : count("initializing") > 0 ? "initializing" : workers && typeof workers === "object" ? "idle" : "unknown";
      return NextResponse.json({ status, ready, scale_to_zero: true, workers: workers ?? null }, { headers });
    }
    if (!process.env.INVOICE_NER_API_URL) return NextResponse.json({ status: "unconfigured", ready: false }, { status: 503, headers });
    const response = await fetch(`${process.env.INVOICE_NER_API_URL}/health`, { cache: "no-store", signal: AbortSignal.timeout(10_000) });
    return NextResponse.json(await response.json(), { status: response.status, headers });
  } catch {
    return NextResponse.json({ status: "unavailable", ready: false }, { status: 502, headers });
  }
}

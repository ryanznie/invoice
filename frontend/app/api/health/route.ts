import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const localApiBaseUrl = process.env.INVOICE_NER_API_URL;
const runpodEndpointId = process.env.RUNPOD_ENDPOINT_ID;
const runpodApiKey = process.env.RUNPOD_API_KEY;
const runpodInvokeBaseUrl =
  process.env.RUNPOD_INVOKE_BASE_URL ?? "https://api.runpod.ai/v2";

export async function GET() {
  try {
    if (runpodEndpointId && runpodApiKey) {
      const response = await fetch(
        `${runpodInvokeBaseUrl}/${runpodEndpointId}/health`,
        {
          headers: { Authorization: `Bearer ${runpodApiKey}` },
          cache: "no-store",
        },
      );
      if (!response.ok) {
        return NextResponse.json(
          { status: "unavailable", scale_to_zero: true },
          { status: 502 },
        );
      }
      const runpod = await response.json();
      return NextResponse.json({
        status: "available",
        scale_to_zero: true,
        workers: runpod.workers ?? null,
      });
    }

    if (!localApiBaseUrl) {
      return NextResponse.json(
        { detail: "No inference backend is configured." },
        { status: 500 },
      );
    }

    const response = await fetch(`${localApiBaseUrl}/health`, {
      cache: "no-store",
    });
    const data = await response.json();

    return NextResponse.json(data, { status: response.status });
  } catch {
    return NextResponse.json(
      { detail: "Unable to reach backend health endpoint." },
      { status: 502 },
    );
  }
}

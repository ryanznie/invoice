import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

const localApiBaseUrl = process.env.INVOICE_NER_API_URL;
const runpodEndpointId = process.env.RUNPOD_ENDPOINT_ID;
const runpodApiKey = process.env.RUNPOD_API_KEY;
const runpodInvokeBaseUrl =
  process.env.RUNPOD_INVOKE_BASE_URL ?? "https://api.runpod.ai/v2";

const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_OCR_BYTES = 2 * 1024 * 1024;
const TERMINAL_STATUSES = new Set([
  "COMPLETED",
  "FAILED",
  "CANCELLED",
  "TIMED_OUT",
]);

type RunpodJob = {
  id?: string;
  status?: string;
  output?: unknown;
  error?: string;
};

function error(detail: string, status: number) {
  return NextResponse.json({ detail }, { status });
}

async function callRunpod(formData: FormData) {
  if (!runpodEndpointId || !runpodApiKey) {
    return error("Runpod is not configured.", 500);
  }

  const image = formData.get("image");
  const ocrFile = formData.get("ocr_file");
  if (!(image instanceof File) || !(ocrFile instanceof File)) {
    return error("An invoice image and OCR TXT/JSON file are required.", 400);
  }
  if (image.size > MAX_IMAGE_BYTES) {
    return error("The invoice image must be 10 MB or smaller.", 413);
  }
  if (ocrFile.size > MAX_OCR_BYTES) {
    return error("The OCR file must be 2 MB or smaller.", 413);
  }

  const input = {
    image_base64: Buffer.from(await image.arrayBuffer()).toString("base64"),
    image_filename: image.name,
    ocr_base64: Buffer.from(await ocrFile.arrayBuffer()).toString("base64"),
    ocr_filename: ocrFile.name,
  };
  const headers = {
    Authorization: `Bearer ${runpodApiKey}`,
    "Content-Type": "application/json",
  };
  const endpointUrl = `${runpodInvokeBaseUrl}/${runpodEndpointId}`;
  const submitted = await fetch(`${endpointUrl}/run`, {
    method: "POST",
    headers,
    body: JSON.stringify({ input }),
    cache: "no-store",
  });
  const job = (await submitted.json()) as RunpodJob;
  if (!submitted.ok || !job.id) {
    console.error("Runpod submission failed", submitted.status, job.status);
    return error("Unable to start invoice processing.", 502);
  }

  const deadline = Date.now() + 240_000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 1_500));
    const statusResponse = await fetch(`${endpointUrl}/status/${job.id}`, {
      headers: { Authorization: `Bearer ${runpodApiKey}` },
      cache: "no-store",
    });
    const status = (await statusResponse.json()) as RunpodJob;

    if (!statusResponse.ok) {
      console.error("Runpod status request failed", statusResponse.status);
      return error("Unable to read invoice processing status.", 502);
    }
    if (!status.status || !TERMINAL_STATUSES.has(status.status)) {
      continue;
    }
    if (status.status === "COMPLETED") {
      return NextResponse.json(status.output, {
        headers: { "Cache-Control": "no-store" },
      });
    }

    console.error("Runpod job failed", status.status, status.error ?? "");
    return error("Invoice processing failed. Please try again.", 502);
  }

  return error("Invoice processing is taking longer than expected.", 504);
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    if (runpodEndpointId && runpodApiKey) {
      return await callRunpod(formData);
    }
    if (!localApiBaseUrl) {
      return error("No inference backend is configured.", 500);
    }

    const response = await fetch(`${localApiBaseUrl}/predict`, {
      method: "POST",
      body: formData,
      cache: "no-store",
    });
    const data = await response.json();

    return NextResponse.json(data, { status: response.status });
  } catch (caught) {
    console.error("Prediction proxy failed", caught);
    return error("Unable to reach the inference backend.", 502);
  }
}

import { NextResponse } from "next/server";
import { setTimeout as delay } from "node:timers/promises";
import { requireAccess } from "@/lib/access";
import { validateUploads, MAX_UPLOAD_BYTES } from "@/lib/upload";
import { validateOcr } from "@/lib/ocr";
import { isResult } from "@/lib/result";

export const dynamic = "force-dynamic";
export const maxDuration = 300;
const terminal = new Set(["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"]);
function processingTimeoutMs() {
  const configured = Number(process.env.INVOICE_PROCESSING_TIMEOUT_MS ?? 240_000);
  return Number.isFinite(configured) ? Math.min(240_000, Math.max(1000, configured)) : 240_000;
}

function error(detail: string, status: number) {
  return NextResponse.json({ detail }, { status, headers: { "Cache-Control": "no-store" } });
}

async function callRunpod(form: FormData, request: Request) {
  const image = form.get("image") as File;
  const ocr = form.get("ocr_file") as File;
  const base = `${process.env.RUNPOD_INVOKE_BASE_URL ?? "https://api.runpod.ai/v2"}/${process.env.RUNPOD_ENDPOINT_ID}`;
  const headers = { Authorization: `Bearer ${process.env.RUNPOD_API_KEY}`, "Content-Type": "application/json" };
  const timeout = processingTimeoutMs();
  const deadline = AbortSignal.timeout(timeout);
  const signal = AbortSignal.any([deadline, request.signal]);
  let jobId: string | undefined;
  let finished = false;
  let response: NextResponse;
  try {
    // No retry: an ambiguous submission may already have created a paid job.
    const submitted = await fetch(`${base}/run`, {
      method: "POST", headers, cache: "no-store", signal,
      body: JSON.stringify({ input: {
        image_base64: Buffer.from(await image.arrayBuffer()).toString("base64"), image_filename: image.name,
        ocr_base64: Buffer.from(await ocr.arrayBuffer()).toString("base64"), ocr_filename: ocr.name,
      }, policy: { executionTimeout: Math.max(5000, timeout), ttl: Math.max(10_000, timeout + 30_000) } }),
    });
    const job = await submitted.json();
    if (!submitted.ok || typeof job?.id !== "string" || !job.id) throw new Error("Submission failed");
    jobId = job.id;
    while (true) {
      const statusResponse = await fetch(`${base}/status/${encodeURIComponent(jobId!)}`, { headers, cache: "no-store", signal });
      const status = await statusResponse.json();
      if (!statusResponse.ok || typeof status?.status !== "string") throw new Error("Invalid job status");
      if (terminal.has(status.status)) {
        finished = true;
        if (status.status !== "COMPLETED") return error("Invoice processing failed. Please try again.", 502);
        if (!isResult(status.output)) return error("The inference service returned an incomplete result.", 502);
        return NextResponse.json(status.output, { headers: { "Cache-Control": "no-store" } });
      }
      if (!["IN_QUEUE", "IN_PROGRESS"].includes(status.status)) throw new Error("Unknown job status");
      await delay(1000, undefined, { signal });
    }
  } catch {
    response = deadline.aborted || request.signal.aborted
      ? error("Invoice processing timed out or was interrupted.", 504)
      : error("Unable to complete invoice processing.", 502);
  }
  // Independent timeout: cancellation must still work after the request aborts.
  if (jobId && !finished) {
    try {
      const cancelled = await fetch(`${base}/cancel/${encodeURIComponent(jobId)}`, {
        method: "POST", headers, cache: "no-store", signal: AbortSignal.timeout(10_000),
      });
      const cancellation = await cancelled.json();
      if (!cancelled.ok || !terminal.has(cancellation?.status)) throw new Error("Cancellation unconfirmed");
    } catch {
      console.error("Unable to confirm Runpod job cancellation", jobId);
      return error("We could not confirm cancellation. The job may still be running; contact the demo owner before retrying.", response.status);
    }
  } else if (!jobId) {
    return error("Unable to confirm job submission. Contact the demo owner before retrying to avoid a duplicate job.", response.status);
  }
  return response;
}

export async function POST(request: Request) {
  const denied = requireAccess(request);
  if (denied) return denied;
  // Vercel also enforces its platform limit; cap streaming bodies locally too.
  const limit = MAX_UPLOAD_BYTES + 64_000;
  if (Number(request.headers.get("content-length")) > limit) return error("Upload is too large.", 413);
  let form: FormData;
  try {
    const reader = request.body?.getReader();
    if (!reader) return error("Upload files using multipart form data.", 400);
    const chunks: Uint8Array[] = [];
    let size = 0;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) { await reader.cancel(); return error("Upload is too large.", 413); }
      chunks.push(value);
    }
    form = await new Response(Buffer.concat(chunks), { headers: { "Content-Type": request.headers.get("content-type") ?? "" } }).formData();
  } catch { return error("Upload files using valid multipart form data.", 400); }
  const image = form.get("image");
  const ocr = form.get("ocr_file");
  if (!(image instanceof File) || !(ocr instanceof File)) return error("A receipt image and matching OCR file are required.", 400);
  const validation = validateUploads(image, ocr) ?? await validateOcr(ocr);
  if (validation) return error(validation, 400);
  if (process.env.RUNPOD_ENDPOINT_ID && process.env.RUNPOD_API_KEY) return callRunpod(form, request);
  if (!process.env.INVOICE_NER_API_URL) return error("No inference backend is configured.", 503);
  const deadline = AbortSignal.timeout(processingTimeoutMs());
  try {
    const response = await fetch(`${process.env.INVOICE_NER_API_URL}/predict`, {
      method: "POST", body: form, cache: "no-store", signal: AbortSignal.any([request.signal, deadline]),
    });
    const data = await response.json();
    if (!response.ok) {
      // Preserve actionable upload errors, but do not expose internal server errors.
      const detail = response.status >= 400 && response.status < 500 && typeof data?.detail === "string" && data.detail.trim()
        ? data.detail
        : "Invoice processing failed. Please check the files and try again.";
      return error(detail, response.status);
    }
    if (!isResult(data)) return error("The inference service returned an incomplete result.", 502);
    return NextResponse.json(data, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return deadline.aborted || request.signal.aborted
      ? error("Invoice processing timed out or was interrupted.", 504)
      : error("Unable to reach the inference backend.", 502);
  }
}

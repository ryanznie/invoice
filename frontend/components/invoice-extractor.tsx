"use client";

import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  FileText,
  ImageIcon,
  Loader2,
  Upload,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const MAX_VISIBLE_PREDICTIONS = 500;
const MAX_VISIBLE_BOXES = 500;

type HealthResponse = {
  status: "healthy" | "unhealthy";
  model_loaded: boolean;
  device: string;
  inference_backend?: "onnx" | "triton";
};

type Prediction = {
  index?: number;
  word: string;
  label: string;
  confidence?: number;
  box?: [number, number, number, number];
  is_invoice_number: boolean;
};

type PredictResponse = {
  invoice_number: string;
  extraction_method: "heuristic" | "model";
  predictions: Prediction[];
  total_words: number;
  model_device: string;
  image_size?: {
    width: number;
    height: number;
  };
};

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function hasBox(prediction: Prediction) {
  return Array.isArray(prediction.box) && prediction.box.length === 4;
}

function fileLabel(file: File | null, fallback: string) {
  return file ? file.name : fallback;
}

function runtimeLabel(health: HealthResponse | null, device?: string) {
  const runtimeDevice = device || health?.device;

  if (!runtimeDevice) {
    return "-";
  }

  if (health?.inference_backend === "triton") {
    return "GPU / Triton";
  }

  if (health?.inference_backend === "onnx") {
    return runtimeDevice === "cpu"
      ? "CPU / ONNX Runtime"
      : `${runtimeDevice.toUpperCase()} / ONNX Runtime`;
  }

  return runtimeDevice;
}

export function InvoiceExtractor() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [ocrFile, setOcrFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthMessage, setHealthMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [requestStatus, setRequestStatus] = useState<string | null>(null);
  const [debugEvents, setDebugEvents] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadHealth() {
      try {
        const response = await fetch("/api/health", {
          signal: controller.signal,
          cache: "no-store",
        });
        const data = (await response.json()) as HealthResponse & { detail?: string };

        if (!response.ok) {
          throw new Error(data.detail || "Health check failed.");
        }

        setHealth(data);
        setHealthMessage(null);
      } catch (healthError) {
        if (controller.signal.aborted) {
          return;
        }

        const message =
          healthError instanceof Error ? healthError.message : "Health check failed.";
        setHealth(null);
        setHealthMessage(message);
        setError(message);
      }
    }

    void loadHealth();

    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!imageFile) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(imageFile);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [imageFile]);

  useEffect(() => {
    if (!isLoading) {
      setElapsedSeconds(0);
      return;
    }

    const startedAt = Date.now();
    const intervalId = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);

    return () => window.clearInterval(intervalId);
  }, [isLoading]);

  const matchedPredictions = useMemo(
    () =>
      result?.predictions.filter((prediction) => prediction.is_invoice_number) ?? [],
    [result],
  );

  const visibleBoxes = useMemo(
    () => result?.predictions.filter(hasBox).slice(0, MAX_VISIBLE_BOXES) ?? [],
    [result],
  );

  const visiblePredictions = useMemo(
    () => result?.predictions.slice(0, MAX_VISIBLE_PREDICTIONS) ?? [],
    [result],
  );

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const debug = (message: string) => {
      const timestamp = new Date().toLocaleTimeString();
      setDebugEvents((events) => [`${timestamp} ${message}`, ...events].slice(0, 8));
    };

    debug("Submit handler fired");

    if (!imageFile || !ocrFile) {
      debug(
        `Blocked: image=${imageFile ? "yes" : "no"}, ocr=${
          ocrFile ? "yes" : "no"
        }`,
      );
      setError("Upload both an invoice image and OCR file.");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("ocr_file", ocrFile);

    setIsLoading(true);
    setRequestStatus("Uploading to local Next proxy");
    debug(`POST /api/predict with ${imageFile.name} and ${ocrFile.name}`);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      debug(`Response ${response.status}`);
      setRequestStatus("Parsing backend response");
      const data = (await response.json()) as PredictResponse & { detail?: string };

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setRequestStatus("Rendering results");
      setIsLoading(false);
      setResult(data);
      debug(`Rendered invoice ${data.invoice_number || "Not Found"}`);
      setRequestStatus(null);
    } catch (submitError) {
      const message =
        submitError instanceof Error ? submitError.message : "Prediction failed.";
      debug(`Error: ${message}`);
      setError(message);
      setIsLoading(false);
      setRequestStatus(null);
    }
  }

  const backendReady = health?.status === "healthy" && health.model_loaded;
  const backendRuntime = runtimeLabel(health);

  return (
    <main className="min-h-screen bg-[hsl(var(--background))] text-foreground">
      <div className="mx-auto grid min-h-screen w-full max-w-7xl grid-rows-[auto_1fr] px-4 py-5 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 border-b border-border pb-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.18em] text-muted-foreground">
              Invoice NER
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
              Extract invoice numbers
            </h1>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-sm">
            <Badge
              variant={backendReady ? "default" : "destructive"}
              className="rounded-md px-2.5 py-1"
            >
              {backendReady ? "Backend ready" : "Backend unavailable"}
            </Badge>
            <span className="text-muted-foreground">
              {health ? backendRuntime : "checking"}
            </span>
          </div>
        </header>

        <p className="mt-3 font-mono text-xs text-muted-foreground">
          API target: local Next proxy -&gt; {health ? backendRuntime : "backend"}
        </p>

        {healthMessage ? (
          <div className="mt-4 border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
            {healthMessage}
          </div>
        ) : null}

        <div className="grid gap-5 py-5 lg:grid-cols-[340px_minmax(0,1fr)]">
          <aside className="space-y-5">
            <form
              className="space-y-4 border border-border bg-card p-4"
              onSubmit={handleSubmit}
            >
              <div className="space-y-2">
                <Label className="flex items-center gap-2" htmlFor="image">
                  <ImageIcon className="h-4 w-4" />
                  Invoice image
                </Label>
                <Input
                  id="image"
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/tiff,image/bmp"
                  onChange={(event) => {
                    setImageFile(event.target.files?.[0] || null);
                    setResult(null);
                  }}
                />
                <p className="truncate text-xs text-muted-foreground">
                  {fileLabel(imageFile, "PNG, JPG, WEBP, TIFF, or BMP")}
                </p>
              </div>

              <div className="space-y-2">
                <Label className="flex items-center gap-2" htmlFor="ocr">
                  <FileText className="h-4 w-4" />
                  OCR payload
                </Label>
                <Input
                  id="ocr"
                  type="file"
                  accept=".txt,.json"
                  onChange={(event) => {
                    setOcrFile(event.target.files?.[0] || null);
                    setResult(null);
                  }}
                />
                <p className="truncate text-xs text-muted-foreground">
                  {fileLabel(ocrFile, ".txt or .json with words and boxes")}
                </p>
              </div>

              <Button className="w-full" type="submit" disabled={isLoading}>
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Upload className="h-4 w-4" />
                )}
                {isLoading ? "Extracting" : "Extract"}
              </Button>

              {isLoading ? (
                <p className="text-center text-xs text-muted-foreground">
                  {requestStatus || "Running inference"} for {elapsedSeconds}s
                </p>
              ) : null}

              {error ? (
                <div className="flex gap-2 border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                  <p>{error}</p>
                </div>
              ) : null}
            </form>

            <section className="space-y-3 border border-border p-4 text-sm">
              <div className="flex items-center justify-between gap-3">
                <span className="text-muted-foreground">Method</span>
                <span className="font-medium capitalize">
                  {result?.extraction_method || "-"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-muted-foreground">Words</span>
                <span className="font-medium">{result?.total_words ?? 0}</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-muted-foreground">Matched</span>
                <span className="font-medium">{matchedPredictions.length}</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-muted-foreground">Runtime</span>
                <span className="font-medium">
                  {runtimeLabel(health, result?.model_device)}
                </span>
              </div>
            </section>

            <details className="border border-border p-4 text-sm">
              <summary className="cursor-pointer font-medium">OCR formats</summary>
              <div className="mt-3 space-y-3 text-muted-foreground">
                <p>TXT: x1,y1,x2,y2,x3,y3,x4,y4,text</p>
                <p>JSON: words with bboxes or boxes.</p>
              </div>
            </details>

            <section className="space-y-2 border border-border p-4 text-xs">
              <p className="font-medium">Debug</p>
              {debugEvents.length ? (
                debugEvents.map((event) => (
                  <p className="break-words font-mono text-muted-foreground" key={event}>
                    {event}
                  </p>
                ))
              ) : (
                <p className="text-muted-foreground">No submit events yet.</p>
              )}
            </section>
          </aside>

          <section className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
            <div className="min-w-0 space-y-5">
              <div className="border border-border bg-card p-4">
                <p className="text-sm text-muted-foreground">Extracted invoice number</p>
                <div className="mt-2 flex min-h-14 items-center gap-3">
                  {result?.invoice_number && result.invoice_number !== "Not Found" ? (
                    <CheckCircle2 className="h-5 w-5 shrink-0 text-primary" />
                  ) : null}
                  <p className="break-all text-3xl font-semibold tracking-tight">
                    {result?.invoice_number || "Waiting for extraction"}
                  </p>
                </div>
              </div>

              <div className="relative min-h-[360px] overflow-hidden border border-border bg-[hsl(var(--preview))]">
                {previewUrl ? (
                  <div className="relative">
                    <img
                      alt="Invoice preview"
                      className="h-auto w-full object-contain"
                      src={previewUrl}
                    />
                    {visibleBoxes.map((prediction, fallbackIndex) => (
                        <div
                          key={`${prediction.index ?? fallbackIndex}-${prediction.word}`}
                          className={
                            prediction.is_invoice_number
                              ? "absolute border-2 border-primary bg-primary/15 shadow-[0_0_0_9999px_rgba(179,38,30,0.02)]"
                              : "absolute border border-[hsl(var(--annotation-muted))]"
                          }
                          style={{
                            left: `${prediction.box![0] / 10}%`,
                            top: `${prediction.box![1] / 10}%`,
                            width: `${
                              (prediction.box![2] - prediction.box![0]) / 10
                            }%`,
                            height: `${
                              (prediction.box![3] - prediction.box![1]) / 10
                            }%`,
                          }}
                          title={`${prediction.word} (${prediction.label})`}
                        />
                      ))}
                  </div>
                ) : (
                  <div className="flex min-h-[360px] items-center justify-center p-8 text-center text-sm text-muted-foreground">
                    Upload an invoice image to preview annotations.
                  </div>
                )}
              </div>
            </div>

            <div className="min-w-0 border border-border bg-card">
              <div className="border-b border-border p-4">
                <h2 className="font-semibold">Word predictions</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Highlighted rows form the selected invoice number.
                  {result && result.predictions.length > visiblePredictions.length
                    ? ` Showing ${visiblePredictions.length} of ${result.predictions.length}.`
                    : ""}
                </p>
              </div>

              <div className="max-h-[680px] overflow-auto">
                {result?.predictions.length ? (
                  <table className="w-full text-left text-sm">
                    <thead className="sticky top-0 bg-card text-xs uppercase text-muted-foreground">
                      <tr className="border-b border-border">
                        <th className="px-3 py-2 font-medium">Word</th>
                        <th className="px-3 py-2 font-medium">Label</th>
                        <th className="px-3 py-2 text-right font-medium">Conf.</th>
                      </tr>
                    </thead>
                    <tbody>
                      {visiblePredictions.map((prediction, fallbackIndex) => (
                        <tr
                          key={`${prediction.index ?? fallbackIndex}-${prediction.word}`}
                          className={
                            prediction.is_invoice_number
                              ? "border-b border-primary/20 bg-primary/10"
                              : "border-b border-border"
                          }
                        >
                          <td className="max-w-36 break-words px-3 py-2 font-medium">
                            {prediction.word}
                          </td>
                          <td className="px-3 py-2 font-mono text-xs">
                            {prediction.label}
                          </td>
                          <td className="px-3 py-2 text-right">
                            {typeof prediction.confidence === "number"
                              ? formatPercent(prediction.confidence)
                              : "-"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="p-8 text-sm text-muted-foreground">
                    No prediction yet.
                  </div>
                )}
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}

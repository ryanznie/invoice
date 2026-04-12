"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, Cpu, FileImage, FileText, ScanSearch, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type HealthResponse = {
  status: "healthy" | "unhealthy";
  model_loaded: boolean;
  device: string;
};

type Prediction = {
  index: number;
  word: string;
  label: string;
  confidence: number;
  box: [number, number, number, number];
  is_invoice_number: boolean;
};

type PredictResponse = {
  invoice_number: string;
  extraction_method: "heuristic" | "model";
  predictions: Prediction[];
  total_words: number;
  model_device: string;
  image_size: {
    width: number;
    height: number;
  };
};

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function InvoiceExtractor() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [ocrFile, setOcrFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadHealth() {
      try {
        const response = await fetch("/api/health", {
          signal: controller.signal,
          cache: "no-store",
        });

        if (!response.ok) {
          const data = (await response.json()) as { detail?: string };
          throw new Error(data.detail || "Health check failed.");
        }

        const data = (await response.json()) as HealthResponse;
        setHealth(data);
      } catch (healthError) {
        const message =
          healthError instanceof Error ? healthError.message : "Health check failed.";
        setHealth(null);
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

  const highlightedCount = useMemo(
    () => result?.predictions.filter((prediction) => prediction.is_invoice_number).length ?? 0,
    [result],
  );

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!imageFile || !ocrFile) {
      setError("Upload both an invoice image and OCR file.");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("ocr_file", ocrFile);

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const data = (await response.json()) as { detail?: string };
        throw new Error(data.detail || "Prediction failed.");
      }

      const data = (await response.json()) as PredictResponse;
      setResult(data);
    } catch (submitError) {
      const message =
        submitError instanceof Error ? submitError.message : "Prediction failed.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen px-4 py-10 sm:px-6">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
        <section className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
          <Card className="overflow-hidden border-none bg-transparent shadow-none">
            <CardHeader className="rounded-[1.5rem] border border-border/70 bg-card/80 backdrop-blur">
              <div className="mb-4 flex items-center gap-3">
                <Badge className="rounded-full px-3 py-1 text-[11px] uppercase tracking-[0.2em]">
                  Vercel Frontend
                </Badge>
                <Badge variant="outline" className="rounded-full px-3 py-1 text-[11px] uppercase tracking-[0.2em]">
                  shadcn/ui
                </Badge>
              </div>
              <CardTitle className="max-w-2xl text-4xl leading-tight sm:text-5xl">
                Replace the Gradio demo with a deployable invoice review surface.
              </CardTitle>
              <CardDescription className="max-w-2xl text-base text-muted-foreground">
                Upload an invoice image and OCR payload, run the hybrid extraction pipeline, and inspect token-level predictions with box overlays.
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="border-border/70 bg-card/85 backdrop-blur">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <ScanSearch className="h-5 w-5 text-primary" />
                Backend Status
              </CardTitle>
              <CardDescription>The frontend proxies requests to your FastAPI deployment through Next.js route handlers.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="flex items-center justify-between rounded-xl bg-secondary/70 p-4">
                <span className="text-muted-foreground">Request Path</span>
                <code className="max-w-[60%] truncate text-right text-xs">/api/health, /api/predict</code>
              </div>
              <div className="flex items-center justify-between rounded-xl bg-secondary/70 p-4">
                <span className="text-muted-foreground">Health</span>
                <Badge variant={health?.status === "healthy" ? "default" : "destructive"}>
                  {health?.status || "unknown"}
                </Badge>
              </div>
              <div className="flex items-center justify-between rounded-xl bg-secondary/70 p-4">
                <span className="text-muted-foreground">Model Loaded</span>
                <span>{health?.model_loaded ? "Yes" : "No"}</span>
              </div>
              <div className="flex items-center justify-between rounded-xl bg-secondary/70 p-4">
                <span className="text-muted-foreground">Device</span>
                <span className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-primary" />
                  {health?.device || "Unavailable"}
                </span>
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 xl:grid-cols-[420px_minmax(0,1fr)]">
          <Card className="bg-card/90 backdrop-blur">
            <CardHeader>
              <CardTitle>Run Extraction</CardTitle>
              <CardDescription>
                Supports invoice images plus OCR data in `.txt` or `.json`.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form className="space-y-5" onSubmit={handleSubmit}>
                <div className="space-y-2">
                  <Label htmlFor="image">Invoice image</Label>
                  <Input
                    id="image"
                    type="file"
                    accept="image/png,image/jpeg,image/webp,image/tiff,image/bmp"
                    onChange={(event) => setImageFile(event.target.files?.[0] || null)}
                  />
                  <p className="text-xs text-muted-foreground">
                    JPG, PNG, TIFF, BMP, or WebP.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="ocr">OCR payload</Label>
                  <Input
                    id="ocr"
                    type="file"
                    accept=".txt,.json"
                    onChange={(event) => setOcrFile(event.target.files?.[0] || null)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Text lines (`x1,y1,x2,y2,x3,y3,x4,y4,text`) or JSON with `words` and `bboxes`.
                  </p>
                </div>

                <div className="grid gap-3 rounded-xl border border-dashed border-border p-4 text-sm">
                  <div className="flex items-center gap-3">
                    <FileImage className="h-4 w-4 text-primary" />
                    <span className="truncate">{imageFile?.name || "No image selected"}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <FileText className="h-4 w-4 text-primary" />
                    <span className="truncate">{ocrFile?.name || "No OCR file selected"}</span>
                  </div>
                </div>

                <Button className="w-full" disabled={isLoading} size="lg" type="submit">
                  {isLoading ? "Extracting..." : "Extract Invoice Number"}
                </Button>

                {isLoading ? <Progress value={68} /> : null}

                {error ? (
                  <div className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
                    <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                    <span>{error}</span>
                  </div>
                ) : null}
              </form>
            </CardContent>
          </Card>

          <div className="grid gap-6">
            <Card className="bg-card/90 backdrop-blur">
              <CardHeader>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <CardTitle>Extraction Result</CardTitle>
                    <CardDescription>
                      The API returns the chosen method, token labels, and normalized boxes.
                    </CardDescription>
                  </div>
                  {result ? (
                    <Badge variant={result.extraction_method === "heuristic" ? "secondary" : "default"}>
                      {result.extraction_method}
                    </Badge>
                  ) : null}
                </div>
              </CardHeader>
              <CardContent className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_320px]">
                <div className="space-y-4">
                  <div className="rounded-2xl border border-border/70 bg-muted/30 p-4">
                    <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">
                      Invoice Number
                    </p>
                    <p className="mt-2 break-all text-3xl font-semibold">
                      {result?.invoice_number || "Waiting for prediction"}
                    </p>
                  </div>

                  <div className="relative overflow-hidden rounded-2xl border border-border/70 bg-slate-950/95">
                    {previewUrl ? (
                      <div className="relative">
                        <img
                          alt="Invoice preview"
                          className="h-auto w-full object-contain"
                          src={previewUrl}
                        />
                        {result?.predictions.map((prediction) => (
                          <div
                            key={`${prediction.index}-${prediction.word}`}
                            className={
                              prediction.is_invoice_number
                                ? "absolute border-2 border-orange-400 bg-orange-400/10"
                                : "absolute border border-cyan-300/70 bg-cyan-300/5"
                            }
                            style={{
                              left: `${prediction.box[0] / 10}%`,
                              top: `${prediction.box[1] / 10}%`,
                              width: `${(prediction.box[2] - prediction.box[0]) / 10}%`,
                              height: `${(prediction.box[3] - prediction.box[1]) / 10}%`,
                            }}
                            title={`${prediction.word} (${prediction.label})`}
                          />
                        ))}
                      </div>
                    ) : (
                      <div className="flex min-h-[420px] items-center justify-center p-10 text-center text-sm text-slate-300">
                        Upload an image to see the box overlay view.
                      </div>
                    )}
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="rounded-2xl bg-secondary/70 p-4">
                    <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">Summary</p>
                    <div className="mt-4 grid gap-3 text-sm">
                      <div className="flex items-center justify-between">
                        <span>Total tokens</span>
                        <span className="font-medium">{result?.total_words ?? 0}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Highlighted tokens</span>
                        <span className="font-medium">{highlightedCount}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Inference device</span>
                        <span className="font-medium">{result?.model_device || health?.device || "Unknown"}</span>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-2xl bg-secondary/70 p-4">
                    <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">Review Notes</p>
                    <ul className="mt-4 space-y-3 text-sm text-muted-foreground">
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="mt-0.5 h-4 w-4 text-chart-3" />
                        Heuristic hits render as orange boxes.
                      </li>
                      <li className="flex items-start gap-2">
                        <Sparkles className="mt-0.5 h-4 w-4 text-chart-2" />
                        Model fallback uses token-level labels and confidence.
                      </li>
                      <li className="flex items-start gap-2">
                        <AlertCircle className="mt-0.5 h-4 w-4 text-chart-4" />
                        Low-confidence rows are easiest to review in the table below.
                      </li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card/90 backdrop-blur">
              <CardHeader>
                <CardTitle>Token Predictions</CardTitle>
                <CardDescription>
                  Review the words the model or heuristic considered part of the invoice number.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>#</TableHead>
                      <TableHead>Word</TableHead>
                      <TableHead>Label</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>Match</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result?.predictions.length ? (
                      result.predictions.map((prediction) => (
                        <TableRow key={`${prediction.index}-${prediction.word}`}>
                          <TableCell className="text-muted-foreground">{prediction.index}</TableCell>
                          <TableCell className="font-medium">{prediction.word}</TableCell>
                          <TableCell>
                            <Badge variant={prediction.is_invoice_number ? "default" : "outline"}>
                              {prediction.label}
                            </Badge>
                          </TableCell>
                          <TableCell>{formatPercent(prediction.confidence)}</TableCell>
                          <TableCell>{prediction.is_invoice_number ? "Yes" : "No"}</TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell className="py-10 text-center text-muted-foreground" colSpan={5}>
                          No prediction yet.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>

                <Separator />

                <div className="grid gap-3 text-sm text-muted-foreground sm:grid-cols-3">
                  <div className="rounded-xl bg-muted/60 p-3">
                    `LABEL_0` means the token is outside the invoice number span.
                  </div>
                  <div className="rounded-xl bg-muted/60 p-3">
                    `LABEL_1` and `LABEL_2` represent the model-selected invoice number span.
                  </div>
                  <div className="rounded-xl bg-muted/60 p-3">
                    `HEURISTIC_MATCH` means the fast rule-based path found the number before model inference.
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </main>
  );
}

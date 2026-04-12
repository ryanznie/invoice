"use client";

import { useEffect, useMemo, useState } from "react";

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

        const data = (await response.json()) as HealthResponse & { detail?: string };

        if (!response.ok) {
          throw new Error(data.detail || "Health check failed.");
        }

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
    () =>
      result?.predictions.filter((prediction) => prediction.is_invoice_number).length ??
      0,
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

      const data = (await response.json()) as PredictResponse & { detail?: string };

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

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
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Invoice Extraction</h1>
        <p className="text-sm text-muted-foreground">
          Upload an invoice image and OCR payload to extract the invoice number.
        </p>
      </div>

      <Card>
        <CardContent className="flex flex-col gap-3 p-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium">Backend status</p>
            <p className="text-sm text-muted-foreground">
              {health
                ? `Model ${health.model_loaded ? "loaded" : "not loaded"} on ${health.device}`
                : "Checking backend"}
            </p>
          </div>
          <Badge
            variant={health?.status === "healthy" ? "default" : "destructive"}
            className="w-fit"
          >
            {health?.status || "unknown"}
          </Badge>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Upload</CardTitle>
            <CardDescription>Supported OCR formats: `.txt` and `.json`.</CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div className="space-y-2">
                <Label htmlFor="image">Invoice image</Label>
                <Input
                  id="image"
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/tiff,image/bmp"
                  onChange={(event) => setImageFile(event.target.files?.[0] || null)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="ocr">OCR payload</Label>
                <Input
                  id="ocr"
                  type="file"
                  accept=".txt,.json"
                  onChange={(event) => setOcrFile(event.target.files?.[0] || null)}
                />
              </div>

              <div className="rounded-md border bg-muted/40 p-3 text-sm">
                <p className="truncate">{imageFile?.name || "No image selected"}</p>
                <p className="mt-1 truncate text-muted-foreground">
                  {ocrFile?.name || "No OCR file selected"}
                </p>
              </div>

              <Button className="w-full" type="submit" disabled={isLoading}>
                {isLoading ? "Extracting..." : "Extract Invoice Number"}
              </Button>

              {isLoading ? <Progress value={70} /> : null}

              {error ? (
                <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                  {error}
                </div>
              ) : null}
            </form>
          </CardContent>
        </Card>

        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle>Result</CardTitle>
                  <CardDescription>
                    Review the extracted invoice number and matching tokens.
                  </CardDescription>
                </div>
                {result ? (
                  <Badge variant="secondary">{result.extraction_method}</Badge>
                ) : null}
              </div>
            </CardHeader>
            <CardContent className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_220px]">
              <div className="space-y-4">
                <div className="rounded-md border p-4">
                  <p className="text-sm text-muted-foreground">Invoice number</p>
                  <p className="mt-2 break-all text-2xl font-semibold">
                    {result?.invoice_number || "Waiting for prediction"}
                  </p>
                </div>

                <div className="relative overflow-hidden rounded-md border bg-muted/20">
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
                              ? "absolute border-2 border-primary bg-primary/10"
                              : "absolute border border-border/70"
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
                    <div className="flex min-h-[320px] items-center justify-center p-6 text-sm text-muted-foreground">
                      Upload an image to preview annotations.
                    </div>
                  )}
                </div>
              </div>

              <div className="space-y-3 rounded-md border p-4 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Total words</span>
                  <span>{result?.total_words ?? 0}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Matched words</span>
                  <span>{highlightedCount}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Device</span>
                  <span>{result?.model_device || health?.device || "Unknown"}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Predictions</CardTitle>
              <CardDescription>
                Token-level labels returned by the backend.
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
                        <TableCell className="text-muted-foreground">
                          {prediction.index}
                        </TableCell>
                        <TableCell className="font-medium">
                          {prediction.word}
                        </TableCell>
                        <TableCell>{prediction.label}</TableCell>
                        <TableCell>{formatPercent(prediction.confidence)}</TableCell>
                        <TableCell>
                          {prediction.is_invoice_number ? "Yes" : "No"}
                        </TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell
                        className="py-10 text-center text-muted-foreground"
                        colSpan={5}
                      >
                        No prediction yet.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>

              <Separator />

              <p className="text-sm text-muted-foreground">
                `LABEL_1`, `LABEL_2`, and `HEURISTIC_MATCH` indicate the selected
                invoice number span.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  );
}

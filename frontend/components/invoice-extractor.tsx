"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { CheckCircle2, FileText, LoaderCircle, ScanLine, Upload } from "lucide-react";

type Prediction = {
  word: string;
  label: string;
  is_invoice_number: boolean;
};

type Result = {
  invoice_number: string;
  extraction_method: string;
  predictions: Prediction[];
  total_words: number;
  model_device: string;
};

export function InvoiceExtractor() {
  const [image, setImage] = useState<File | null>(null);
  const [ocrFile, setOcrFile] = useState<File | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState("checking");

  const previewUrl = useMemo(
    () => (image ? URL.createObjectURL(image) : null),
    [image],
  );

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    fetch("/api/health", { cache: "no-store" })
      .then((response) => {
        setBackendStatus(response.ok ? "available" : "unavailable");
      })
      .catch(() => setBackendStatus("unavailable"));
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!image || !ocrFile) {
      setError("Choose both an invoice image and its coordinate-bearing TXT or JSON file.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("image", image);
    formData.append("ocr_file", ocrFile);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Invoice processing failed.");
      }
      setResult(data as Result);
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Invoice processing failed.",
      );
    } finally {
      setLoading(false);
    }
  }

  const matchedWords = result?.predictions.filter(
    (prediction) => prediction.is_invoice_number,
  );

  return (
    <main className="min-h-screen bg-slate-50 px-4 py-10 text-slate-950 sm:px-6">
      <div className="mx-auto max-w-5xl">
        <header className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-blue-700">
              Document intelligence
            </p>
            <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              Invoice number extractor
            </h1>
            <p className="mt-3 max-w-2xl text-slate-600">
              Upload an invoice and its coordinate-bearing OCR file. Processing runs on a
              private inference worker and files are not retained.
            </p>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                backendStatus === "available"
                  ? "bg-emerald-500"
                  : backendStatus === "checking"
                    ? "bg-amber-400"
                    : "bg-rose-500"
              }`}
            />
            {backendStatus === "available"
              ? "Backend available"
              : backendStatus === "checking"
                ? "Checking backend"
                : "Backend unavailable"}
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <form
            onSubmit={submit}
            className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
          >
            <div className="mb-6 flex items-center gap-3">
              <div className="rounded-xl bg-blue-50 p-2.5 text-blue-700">
                <ScanLine className="h-5 w-5" />
              </div>
              <div>
                <h2 className="font-semibold">New extraction</h2>
                <p className="text-sm text-slate-500">JPG/PNG + TXT/JSON</p>
              </div>
            </div>

            <label className="mb-4 block cursor-pointer rounded-xl border border-dashed border-slate-300 p-4 transition hover:border-blue-500 hover:bg-blue-50/40">
              <span className="flex items-center gap-3">
                <Upload className="h-5 w-5 text-slate-500" />
                <span>
                  <span className="block text-sm font-medium">Invoice image</span>
                  <span className="block text-xs text-slate-500">Maximum 10 MB</span>
                </span>
              </span>
              <input
                className="sr-only"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(event) => setImage(event.target.files?.[0] ?? null)}
              />
              {image && <span className="mt-3 block truncate text-sm text-blue-700">{image.name}</span>}
            </label>

            <label className="mb-6 block cursor-pointer rounded-xl border border-dashed border-slate-300 p-4 transition hover:border-blue-500 hover:bg-blue-50/40">
              <span className="flex items-center gap-3">
                <FileText className="h-5 w-5 text-slate-500" />
                <span>
                  <span className="block text-sm font-medium">OCR coordinates</span>
                  <span className="block text-xs text-slate-500">TXT or JSON, maximum 2 MB</span>
                </span>
              </span>
              <input
                className="sr-only"
                type="file"
                accept=".txt,.json,text/plain,application/json"
                onChange={(event) => setOcrFile(event.target.files?.[0] ?? null)}
              />
              {ocrFile && (
                <span className="mt-3 block truncate text-sm text-blue-700">{ocrFile.name}</span>
              )}
            </label>

            {error && (
              <p className="mb-4 rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading || !image || !ocrFile}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <>
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                  Waking worker and processing…
                </>
              ) : (
                "Extract invoice number"
              )}
            </button>
            <p className="mt-3 text-center text-xs text-slate-500">
              The first request after an idle period may take longer while the CPU worker starts.
            </p>
          </form>

          <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 p-6">
              <h2 className="font-semibold">Result</h2>
              <p className="mt-1 text-sm text-slate-500">Review the extracted value before use.</p>
            </div>

            {previewUrl && (
              <div className="border-b border-slate-200 bg-slate-100 p-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewUrl}
                  alt="Invoice preview"
                  className="mx-auto max-h-72 rounded-lg object-contain"
                />
              </div>
            )}

            <div className="p-6">
              {result ? (
                <div>
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="mt-1 h-5 w-5 shrink-0 text-emerald-600" />
                    <div>
                      <p className="text-sm text-slate-500">Invoice number</p>
                      <p className="mt-1 break-all text-2xl font-semibold tracking-tight">
                        {result.invoice_number}
                      </p>
                    </div>
                  </div>
                  <dl className="mt-6 grid grid-cols-2 gap-4 rounded-xl bg-slate-50 p-4 text-sm">
                    <div>
                      <dt className="text-slate-500">Method</dt>
                      <dd className="mt-1 font-medium capitalize">{result.extraction_method}</dd>
                    </div>
                    <div>
                      <dt className="text-slate-500">Words processed</dt>
                      <dd className="mt-1 font-medium">{result.total_words}</dd>
                    </div>
                  </dl>
                  {matchedWords && matchedWords.length > 0 && (
                    <div className="mt-5">
                      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                        Matched tokens
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {matchedWords.map((prediction, index) => (
                          <span
                            key={`${prediction.word}-${index}`}
                            className="rounded-full bg-blue-50 px-3 py-1 text-sm text-blue-800"
                          >
                            {prediction.word}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="py-12 text-center text-sm text-slate-500">
                  Your extraction result will appear here.
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}

"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { ArrowRight, Check, CheckCircle2, Copy, FileText, ImagePlus, LoaderCircle, ReceiptText, RotateCcw, ScanLine, ZoomIn, ZoomOut } from "lucide-react";
import { validateUploads } from "@/lib/upload";
import { readOcrWordBoxes, type OcrWordBox } from "@/lib/ocr";

import { isResult, type Result } from "@/lib/result";

export function InvoiceExtractor() {
  const [image, setImage] = useState<File | null>(null);
  const [ocrFile, setOcrFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [value, setValue] = useState("");
  const [reviewed, setReviewed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [zoomed, setZoomed] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [copyStatus, setCopyStatus] = useState("");
  const [fileKey, setFileKey] = useState(0);
  const request = useRef<AbortController | null>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    if (!image) { setPreviewUrl(null); return; }
    const url = URL.createObjectURL(image);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);
  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => { if (result) resultHeading.current?.focus(); }, [result]);

  function clearResult() {
    setResult(null); setValue(""); setReviewed(false); setError(null); setCopyStatus("");
  }

  function chooseImage(file?: File) {
    if (!file || loading) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setError("Choose a JPG, PNG, or WebP receipt image."); return;
    }
    if (file.size === 0 || file.size > 4_000_000) {
      setError("Choose a non-empty receipt image smaller than 4 MB."); return;
    }
    clearResult(); setImage(file); setOcrFile(null); setZoomed(false); setFileKey((key) => key + 1);
  }

  function reset() {
    clearResult(); setImage(null); setOcrFile(null); setZoomed(false); setFileKey((key) => key + 1);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (loading || !image || !ocrFile) return;
    const validation = validateUploads(image, ocrFile);
    if (validation) { setError(validation); return; }
    clearResult(); setLoading(true);
    const controller = new AbortController();
    request.current = controller;
    const timeout = window.setTimeout(() => controller.abort(), 270_000);
    const form = new FormData();
    form.append("image", image); form.append("ocr_file", ocrFile);
    try {
      const ocrBoxesPromise: Promise<OcrWordBox[]> = readOcrWordBoxes(ocrFile, image).catch(() => []);
      const response = await fetch("/api/predict", { method: "POST", body: form, signal: controller.signal });
      if (response.status === 413) throw new Error("These files are too large. Try a smaller receipt image.");
      const data = await response.json().catch(() => null);
      if (!response.ok) throw new Error(typeof data?.detail === "string" ? data.detail : "We couldn’t read this receipt. Please try again.");
      if (!isResult(data)) throw new Error("We received an incomplete result. Please try again.");
      const ocrBoxes = await ocrBoxesPromise;
      const predictions = data.predictions.map((prediction, index) => {
        const ocrBox = ocrBoxes[index];
        const alignedBox = ocrBox?.word === prediction.word ? ocrBox.bbox : null;
        return { ...prediction, bbox: prediction.bbox ?? alignedBox };
      });
      setResult({ ...data, predictions });
      setValue(data.invoice_number === "Not Found" ? "" : data.invoice_number);
    } catch (caught) {
      setError(controller.signal.aborted ? "This is taking longer than expected. Please try again." : caught instanceof Error ? caught.message : "Unable to process the receipt. Please try again.");
    } finally {
      window.clearTimeout(timeout); request.current = null; setLoading(false);
    }
  }

  async function copyValue() {
    try { await navigator.clipboard.writeText(value.trim()); setCopyStatus("Copied to clipboard"); }
    catch { setCopyStatus("Couldn’t copy. Select the number and copy it manually."); }
  }

  const found = !!result?.invoice_number.trim() && result.invoice_number !== "Not Found";
  const matches = result?.predictions.filter((item) => item.is_invoice_number) ?? [];
  const boxedMatches = matches.filter((item) => item.bbox && item.bbox[2] > item.bbox[0] && item.bbox[3] > item.bbox[1]);

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="Document review home"><span className="brand-icon"><ReceiptText size={21} /></span>Document review</a>
      </header>
      <main className="workspace">
        <div className="intro">
          <h1>Invoice review</h1>
          <p>Upload a receipt or invoice. Verify the extracted number.</p>
        </div>
        <div className="review-grid">
          <section className="panel receipt-panel" aria-labelledby="receipt-heading">
            <div className="panel-heading"><div><h2 id="receipt-heading">Source document</h2></div>{image && <button className="text-button" type="button" onClick={reset} disabled={loading}><RotateCcw size={14} /> Start over</button>}</div>
            <form onSubmit={submit}>
              <fieldset disabled={loading}>
                {!previewUrl ? (
                  <label className={`dropzone ${dragging ? "dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); chooseImage(event.dataTransfer.files[0]); }}>
                    <span className="upload-art"><ReceiptText size={38} strokeWidth={1.3} /><span><ImagePlus size={17} /></span></span>
                    <strong>Upload document</strong><span>Drop an image here or click to browse</span><span className="file-types">JPG, PNG or WebP · up to 4 MB total</span>
                    <input key={fileKey} aria-label="Receipt image" className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => chooseImage(event.target.files?.[0])} />
                  </label>
                ) : (
                  <div className="preview-wrap">
                    <div className="preview-toolbar"><span title={image?.name}>{image?.name}</span><div className="preview-tools">{boxedMatches.length > 0 && <span className="match-legend"><span aria-hidden="true" />Invoice number</span>}<button type="button" className="text-button" aria-pressed={zoomed} onClick={() => setZoomed(!zoomed)}>{zoomed ? <ZoomOut size={16} /> : <ZoomIn size={16} />}{zoomed ? "Fit" : "Zoom"}</button></div></div>
                    <div className={`receipt-preview ${zoomed ? "zoomed" : ""}`} tabIndex={0} aria-label="Receipt preview; scroll to inspect">
                      <div className="receipt-image-frame">
                        {/* User-selected local image; must retain its original detail. */}
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={previewUrl} alt="Your uploaded receipt" onError={() => { setImage(null); clearResult(); setError("This image could not be opened. Try a different JPG, PNG, or WebP file."); }} />
                        {boxedMatches.length > 0 && <svg className="detection-overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none" role="img" aria-label={`${boxedMatches.length} detected invoice number ${boxedMatches.length === 1 ? "word" : "words"} highlighted in red`}>
                          {boxedMatches.map((item, index) => {
                            const [x0, y0, x1, y1] = item.bbox!;
                            return <rect key={`${item.word}-${index}`} x={x0} y={y0} width={x1 - x0} height={y1 - y0}><title>{item.word}</title></rect>;
                          })}
                        </svg>}
                      </div>
                    </div>
                  </div>
                )}
                <div className="upload-controls">
                  <label className="ocr-label" htmlFor="ocr-file"><FileText size={17} /><span>OCR file <span className="required">Required</span></span></label>
                  <p className="help-text" id="ocr-help">Matching TXT or JSON with text coordinates.</p>
                  <input key={`ocr-${fileKey}`} id="ocr-file" aria-describedby="ocr-help" className="file-input" type="file" accept=".txt,.json" onChange={(event) => { clearResult(); setOcrFile(event.target.files?.[0] ?? null); }} />
                  <details className="format-help"><summary>Which file do I need?</summary><p>A TXT file with text and corner coordinates, or JSON with <code>words</code> and <code>bboxes</code> (or <code>boxes</code>). Plain text alone won’t work. OCR file: up to 2 MB; both files: up to 4 MB combined.</p></details>
                  {error && <p className="error-message" role="alert">{error}</p>}
                  <button className="primary-button" type="submit" disabled={loading || !image || !ocrFile}>{loading ? <><LoaderCircle size={17} className="animate-spin" />Extracting…</> : <>Extract invoice number<ArrowRight size={17} /></>}</button>
                  <p className="under-button" role="status">{loading ? "This may take a few minutes on the first request. Keep this page open." : ""}</p>
                </div>
              </fieldset>
            </form>
          </section>

          <section className={`panel result-panel ${reviewed ? "is-reviewed" : ""}`} aria-labelledby="result-heading" aria-busy={loading}>
            <div className="panel-heading"><div><h2 id="result-heading" ref={resultHeading} tabIndex={-1}>Extracted data</h2></div>{result && <span className={`status-badge ${reviewed ? "reviewed" : ""}`}>{reviewed ? <Check size={13} /> : <span className="status-dot" />}{reviewed ? "Reviewed" : "Needs review"}</span>}</div>
            {result ? (
              <div className="result-content">
                <div className={`review-note ${reviewed ? "success" : ""}`}><span>{reviewed ? <CheckCircle2 size={19} /> : <ScanLine size={19} />}</span><div><strong>{reviewed ? "Number confirmed" : found ? "Verify against the source document" : "We couldn’t find an invoice number."}</strong><p>{reviewed ? "Ready to copy." : found ? "Correct the value below if needed." : "Check the receipt and enter the number yourself, or try a clearer image."}</p></div></div>
                <label className="value-label" htmlFor="invoice-number">Invoice number</label>
                <input id="invoice-number" className="invoice-value" placeholder="Enter the number from your receipt" value={value} onChange={(event) => { setValue(event.target.value); setReviewed(false); setCopyStatus(""); }} spellCheck={false} autoComplete="off" />
                {found && value !== result.invoice_number && <p className="help-text original-value">Originally extracted: <span>{result.invoice_number}</span></p>}
                {matches.length > 1 && !reviewed && <p className="multiple-note">More than one word was matched. Check that they belong to the same invoice number.</p>}
                <div className="review-actions"><button className="primary-button" type="button" disabled={!value.trim() || reviewed} onClick={() => { setValue(value.trim()); setReviewed(true); }}><Check size={17} />{reviewed ? "Confirmed" : "Confirm number"}</button><button className="secondary-button" type="button" disabled={!value.trim()} onClick={copyValue}><Copy size={16} />Copy number</button></div>
                <p className="local-note" role="status">{copyStatus || "Edits and review status stay on this page only."}</p>
                <details className="extraction-details"><summary>Extraction details<span>{result.total_words} words read</span></summary><p className="help-text">Method: {result.extraction_method}</p><p className="help-text">Matched words are highlighted below and boxed on the source document.</p><div className="word-list">{result.predictions.map((item, index) => <span key={index} className={item.is_invoice_number ? "matched" : ""}>{item.word}</span>)}</div></details>
                <button className="next-receipt text-button" type="button" onClick={reset}>New document<ArrowRight size={15} /></button>
              </div>
            ) : (
              <div className="empty-result" role="status"><span className={`empty-icon ${loading ? "processing" : ""}`}>{loading ? <LoaderCircle size={27} className="animate-spin" /> : <ScanLine size={29} strokeWidth={1.4} />}</span><h3>{loading ? "Extracting invoice number…" : "No data extracted"}</h3><p>{loading ? "This may take a few minutes." : "Upload a document and OCR file to begin."}</p></div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

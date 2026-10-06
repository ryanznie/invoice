import type { BoundingBox } from "@/lib/result";

export type OcrWordBox = { word: string; bbox: BoundingBox };

type OcrEntry = { text: string; bbox: [number, number, number, number] };

function splitWords(text: string): string[] {
  const tokens: string[] = [];
  for (const part of text.trim().split(/([\/:.#()\[\]-])/)) {
    if (!part) continue;
    if (/^[\/:.#()\[\]-]$/.test(part)) tokens.push(part);
    else tokens.push(...part.split(/\s+/).filter(Boolean));
  }
  return tokens;
}

function tokenBoxes(text: string, tokens: string[], box: [number, number, number, number]): [number, number, number, number][] {
  if (tokens.length === 1) return [box];
  const [x0, y0, x1, y1] = box;
  const width = x1 - x0;
  const boxes: [number, number, number, number][] = [];
  let charPosition = 0;
  const totalChars = Array.from(text).length;
  for (const token of tokens) {
    const start = text.indexOf(token, charPosition);
    let tokenX0: number;
    let tokenX1: number;
    if (start < 0) {
      const tokenWidth = width / tokens.length;
      tokenX0 = x0 + boxes.length * tokenWidth;
      tokenX1 = tokenX0 + tokenWidth;
    } else {
      const end = start + token.length;
      const startChar = Array.from(text.slice(0, start)).length;
      const endChar = Array.from(text.slice(0, end)).length;
      tokenX0 = x0 + (startChar / Math.max(totalChars, 1)) * width;
      tokenX1 = x0 + (endChar / Math.max(totalChars, 1)) * width;
      charPosition = end;
    }
    boxes.push([Math.trunc(tokenX0), y0, Math.trunc(tokenX1), y1]);
  }
  return boxes;
}

function normalizedBox(box: [number, number, number, number], width: number, height: number): BoundingBox {
  const normalized: BoundingBox = [
    Math.max(0, Math.min(1000, Math.trunc((box[0] * 1000) / width))),
    Math.max(0, Math.min(1000, Math.trunc((box[1] * 1000) / height))),
    Math.max(0, Math.min(1000, Math.trunc((box[2] * 1000) / width))),
    Math.max(0, Math.min(1000, Math.trunc((box[3] * 1000) / height))),
  ];
  if (normalized[0] >= normalized[2]) {
    if (normalized[2] < 1000) normalized[2] = normalized[0] + 1;
    else normalized[0] = Math.max(0, normalized[2] - 1);
  }
  if (normalized[1] >= normalized[3]) {
    if (normalized[3] < 1000) normalized[3] = normalized[1] + 1;
    else normalized[1] = Math.max(0, normalized[3] - 1);
  }
  return normalized;
}

function imageDimensions(file: File): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve({ width: image.naturalWidth, height: image.naturalHeight });
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not read image dimensions"));
    };
    image.src = url;
  });
}

/** Parse OCR token boxes using the same word ordering and splitting as the API. */
export async function readOcrWordBoxes(file: File, image: File): Promise<OcrWordBox[]> {
  const text = await file.text();
  let words: string[];
  let boxes: [number, number, number, number][];
  let pixelCoordinates = false;

  if (file.name.toLowerCase().endsWith(".json")) {
    const data = JSON.parse(text);
    words = data.words;
    boxes = data.bboxes ?? data.boxes;
    pixelCoordinates = boxes.some((box: number[]) => box.some((coordinate) => coordinate > 1000));
    if (!Array.isArray(words) || !Array.isArray(boxes) || words.length !== boxes.length) return [];
  } else {
    const entries: OcrEntry[] = [];
    for (const line of text.split(/\r?\n/)) {
      const parts = line.trim().split(",");
      if (parts.length < 9 || !parts.slice(0, 8).every((part) => /^[+-]?\d+$/.test(part.trim()))) continue;
      const coordinates = parts.slice(0, 8).map((part) => Number(part.trim()));
      const lineText = parts.slice(8).join(",").trim();
      if (!lineText) continue;
      const xs = coordinates.filter((_, index) => index % 2 === 0);
      const ys = coordinates.filter((_, index) => index % 2 === 1);
      entries.push({
        text: lineText,
        bbox: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
      });
    }
    entries.sort((left, right) => left.bbox[1] - right.bbox[1] || left.bbox[0] - right.bbox[0]);
    words = [];
    boxes = [];
    for (const entry of entries) {
      const tokens = splitWords(entry.text);
      words.push(...tokens);
      boxes.push(...tokenBoxes(entry.text, tokens, entry.bbox));
    }
    pixelCoordinates = true;
  }

  if (pixelCoordinates) {
    const dimensions = await imageDimensions(image);
    if (!dimensions.width || !dimensions.height) return [];
    boxes = boxes.map((box) => normalizedBox(box, dimensions.width, dimensions.height));
  }

  return words.map((word, index) => ({ word, bbox: boxes[index] }));
}

export async function validateOcr(file: File): Promise<string | null> {
  let text: string;
  try { text = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer()); }
  catch { return "The OCR file must contain valid UTF-8 text."; }
  const coordinate = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n) && n >= 0;
  if (file.name.toLowerCase().endsWith(".json")) {
    let data;
    try { data = JSON.parse(text); } catch { return "The OCR file contains invalid JSON."; }
    if (!data || typeof data !== "object" || Array.isArray(data)) return "OCR JSON must be an object.";
    const { words, ocr_lines: lines } = data;
    const boxes = data.bboxes ?? data.boxes;
    if (!Array.isArray(words) || !words.length || !words.every((word) => typeof word === "string" && word.trim())) {
      return "OCR JSON must contain a non-empty array of words.";
    }
    if (!Array.isArray(boxes) || boxes.length !== words.length || !boxes.every((box) =>
      Array.isArray(box) && box.length === 4 && box.every(coordinate) && box[0] <= box[2] && box[1] <= box[3])) {
      return "OCR JSON needs one valid [x0, y0, x1, y1] box per word.";
    }
    if (lines !== undefined && lines !== null && (!Array.isArray(lines) || !lines.every((line) => typeof line === "string"))) {
      return "OCR lines must be an array of strings.";
    }
  } else {
    const lines = text.split(/\r?\n/).filter((line) => line.trim());
    if (!lines.length) return "The OCR file contains no text lines.";
    let usableLines = 0;
    for (const line of lines) {
      const parts = line.split(",");
      // Match parse_ocr_text_file: short lines are ignored, not fatal.
      if (parts.length < 9) continue;
      if (!parts.slice(0, 8).every((part) => /^[+-]?\d+$/.test(part.trim()) && Number.isSafeInteger(Number(part)))) {
        return "OCR TXT coordinates must be integers.";
      }
      if (!parts.slice(8).join(",").trim()) continue;
      usableLines++;
    }
    if (!usableLines) return "The OCR file contains no usable text and coordinate lines.";
  }
  return null;
}

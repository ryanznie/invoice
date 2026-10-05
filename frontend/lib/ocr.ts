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

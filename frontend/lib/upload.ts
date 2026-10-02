// Leave room for multipart headers below Vercel's 4.5 MB request limit.
export const MAX_UPLOAD_BYTES = 4_000_000;
export const MAX_OCR_BYTES = 2_000_000;

export function validateUploads(image: File, ocr: File): string | null {
  if (!image.size || !ocr.size) return "One of your files is empty. Choose another file.";
  if (!["image/jpeg", "image/png", "image/webp"].includes(image.type)) {
    return "Choose a JPG, PNG, or WebP receipt image.";
  }
  if (!/\.(txt|json)$/i.test(ocr.name)) return "Choose a TXT or JSON OCR file.";
  if (ocr.size > MAX_OCR_BYTES) return "The OCR file must be 2 MB or smaller.";
  if (image.size + ocr.size > MAX_UPLOAD_BYTES) {
    return "Your files must be 4 MB or smaller in total. Try a smaller receipt image.";
  }
  return null;
}

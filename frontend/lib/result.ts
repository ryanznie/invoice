export type Prediction = { word: string; label: string; is_invoice_number: boolean };
export type Result = { invoice_number: string; extraction_method: string; predictions: Prediction[]; total_words: number };
export function isResult(value: unknown): value is Result {
  if (!value || typeof value !== "object") return false;
  const data = value as Result;
  return typeof data.invoice_number === "string" && typeof data.extraction_method === "string" &&
    Number.isSafeInteger(data.total_words) && data.total_words >= 0 && Array.isArray(data.predictions) && data.predictions.every(
      (item) => item && typeof item.word === "string" && typeof item.label === "string" && typeof item.is_invoice_number === "boolean",
    );
}

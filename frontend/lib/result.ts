export type BoundingBox = [number, number, number, number];
export type Prediction = { word: string; label: string; is_invoice_number: boolean; bbox?: BoundingBox | null };
export type Result = { invoice_number: string; extraction_method: string; predictions: Prediction[]; total_words: number };
function isBoundingBox(value: unknown): value is BoundingBox {
  return Array.isArray(value) && value.length === 4 && value.every(
    (coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate) && coordinate >= 0 && coordinate <= 1000,
  ) && value[0] <= value[2] && value[1] <= value[3];
}
export function isResult(value: unknown): value is Result {
  if (!value || typeof value !== "object") return false;
  const data = value as Result;
  return typeof data.invoice_number === "string" && typeof data.extraction_method === "string" &&
    Number.isSafeInteger(data.total_words) && data.total_words >= 0 && Array.isArray(data.predictions) && data.predictions.every(
      (item) => item && typeof item.word === "string" && typeof item.label === "string" && typeof item.is_invoice_number === "boolean" &&
        (item.bbox === undefined || item.bbox === null || isBoundingBox(item.bbox)),
    );
}

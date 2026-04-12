import { NextResponse } from "next/server";

const apiBaseUrl = process.env.INVOICE_NER_API_URL;

export async function POST(request: Request) {
  if (!apiBaseUrl) {
    return NextResponse.json(
      { detail: "INVOICE_NER_API_URL is not configured." },
      { status: 500 },
    );
  }

  try {
    const formData = await request.formData();
    const response = await fetch(`${apiBaseUrl}/predict`, {
      method: "POST",
      body: formData,
      cache: "no-store",
    });
    const data = await response.json();

    return NextResponse.json(data, { status: response.status });
  } catch {
    return NextResponse.json(
      { detail: "Unable to reach backend predict endpoint." },
      { status: 502 },
    );
  }
}

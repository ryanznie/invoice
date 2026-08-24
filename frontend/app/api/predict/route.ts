import { NextResponse } from "next/server";

const apiBaseUrl = process.env.INVOICE_NER_API_URL;

async function readBackendResponse(response: Response) {
  const text = await response.text();

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return {
      detail: `Backend predict returned ${response.status}: ${
        text || response.statusText
      }`,
    };
  }
}

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
    const data = await readBackendResponse(response);

    return NextResponse.json(data, { status: response.status });
  } catch {
    return NextResponse.json(
      { detail: "Unable to reach backend predict endpoint." },
      { status: 502 },
    );
  }
}

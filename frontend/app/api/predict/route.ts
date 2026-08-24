import { NextResponse } from "next/server";

const apiBaseUrl = process.env.INVOICE_NER_API_URL;

type MultipartFile = {
  contentType: string;
  data: Buffer;
  filename: string;
};

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

function getBoundary(contentType: string | null) {
  const match = contentType?.match(/boundary=(?:"([^"]+)"|([^;]+))/i);
  return match?.[1] || match?.[2] || null;
}

function parseContentDisposition(value: string) {
  const fields = new Map<string, string>();

  for (const part of value.split(";")) {
    const [rawKey, ...rawValue] = part.trim().split("=");
    if (!rawKey || rawValue.length === 0) {
      continue;
    }

    fields.set(rawKey.toLowerCase(), rawValue.join("=").replace(/^"|"$/g, ""));
  }

  return fields;
}

function parseMultipartFiles(body: Buffer, boundary: string) {
  const files = new Map<string, MultipartFile>();
  const boundaryBuffer = Buffer.from(`--${boundary}`);
  let cursor = 0;

  while (cursor < body.length) {
    const boundaryStart = body.indexOf(boundaryBuffer, cursor);
    if (boundaryStart === -1) {
      break;
    }

    let partStart = boundaryStart + boundaryBuffer.length;
    if (body.subarray(partStart, partStart + 2).equals(Buffer.from("--"))) {
      break;
    }

    if (body.subarray(partStart, partStart + 2).equals(Buffer.from("\r\n"))) {
      partStart += 2;
    }

    const headerEnd = body.indexOf(Buffer.from("\r\n\r\n"), partStart);
    if (headerEnd === -1) {
      break;
    }

    const headerText = body.subarray(partStart, headerEnd).toString("utf8");
    const headers = new Map<string, string>();
    for (const line of headerText.split("\r\n")) {
      const separator = line.indexOf(":");
      if (separator === -1) {
        continue;
      }

      headers.set(
        line.slice(0, separator).trim().toLowerCase(),
        line.slice(separator + 1).trim(),
      );
    }

    const dataStart = headerEnd + 4;
    const nextBoundary = body.indexOf(boundaryBuffer, dataStart);
    if (nextBoundary === -1) {
      break;
    }

    const dataEnd =
      nextBoundary >= 2 &&
      body.subarray(nextBoundary - 2, nextBoundary).equals(Buffer.from("\r\n"))
        ? nextBoundary - 2
        : nextBoundary;
    const disposition = headers.get("content-disposition");

    if (disposition) {
      const dispositionFields = parseContentDisposition(disposition);
      const name = dispositionFields.get("name");
      const filename = dispositionFields.get("filename");

      if (name && filename) {
        files.set(name, {
          contentType: headers.get("content-type") || "application/octet-stream",
          data: body.subarray(dataStart, dataEnd),
          filename,
        });
      }
    }

    cursor = nextBoundary;
  }

  return files;
}

function toArrayBuffer(buffer: Buffer) {
  return buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  ) as ArrayBuffer;
}

export async function POST(request: Request) {
  if (!apiBaseUrl) {
    return NextResponse.json(
      { detail: "INVOICE_NER_API_URL is not configured." },
      { status: 500 },
    );
  }

  try {
    const contentType = request.headers.get("content-type");
    console.info(
      `[predict proxy] route entered content-length=${request.headers.get(
        "content-length",
      )} content-type=${contentType}`,
    );
    const boundary = getBoundary(contentType);

    if (!boundary) {
      return NextResponse.json(
        { detail: "Predict upload must be multipart/form-data." },
        { status: 400 },
      );
    }

    const body = Buffer.from(await request.arrayBuffer());
    const files = parseMultipartFiles(body, boundary);
    const image = files.get("image");
    const ocrFile = files.get("ocr_file");

    if (!image || !ocrFile) {
      return NextResponse.json(
        { detail: "Upload both an invoice image and OCR file." },
        { status: 400 },
      );
    }

    console.info(
      `[predict proxy] received image=${image.filename} (${image.data.length} bytes), ocr_file=${ocrFile.filename} (${ocrFile.data.length} bytes)`,
    );

    const outgoingFormData = new FormData();
    outgoingFormData.append(
      "image",
      new Blob([toArrayBuffer(image.data)], { type: image.contentType }),
      image.filename,
    );
    outgoingFormData.append(
      "ocr_file",
      new Blob([toArrayBuffer(ocrFile.data)], { type: ocrFile.contentType }),
      ocrFile.filename,
    );

    const targetUrl = `${apiBaseUrl}/predict`;
    console.info(`[predict proxy] forwarding to ${targetUrl}`);
    const response = await fetch(targetUrl, {
      method: "POST",
      body: outgoingFormData,
      cache: "no-store",
    });
    console.info(`[predict proxy] backend responded ${response.status}`);
    const data = await readBackendResponse(response);

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("[predict proxy] failed", error);
    return NextResponse.json(
      { detail: "Unable to reach backend predict endpoint." },
      { status: 502 },
    );
  }
}

import { createHash, timingSafeEqual } from "node:crypto";
import { NextResponse } from "next/server";

// Vercel Production is public; Preview and local production builds keep the demo gate.
export function requireAccess(request: Request): NextResponse | null {
  if (process.env.VERCEL_ENV !== "production") {
    const password = process.env.DEMO_PASSWORD;
    if (!password && process.env.NODE_ENV === "development" && !process.env.VERCEL && !process.env.RUNPOD_API_KEY) return null;
    if (!password || password.length < 32) {
      return NextResponse.json({ detail: "Demo access is not configured. Set DEMO_PASSWORD to at least 32 characters." }, { status: 503 });
    }
    const authorization = request.headers.get("authorization") ?? "";
    const expected = `Basic ${Buffer.from(`demo:${password}`).toString("base64")}`;
    const hash = (value: string) => createHash("sha256").update(value).digest();
    if (authorization.length > 1024 || !timingSafeEqual(hash(authorization), hash(expected))) {
      return NextResponse.json({ detail: "Sign in to use this demo." }, {
        status: 401,
        headers: { "WWW-Authenticate": 'Basic realm="Invoice review", charset="UTF-8"', "Cache-Control": "no-store" },
      });
    }
  }
  if (request.method === "POST") {
    const origin = request.headers.get("origin");
    if ((origin && origin !== new URL(request.url).origin) || request.headers.get("sec-fetch-site") === "cross-site") {
      return NextResponse.json({ detail: "Cross-site submissions are not allowed." }, { status: 403 });
    }
  }
  return null;
}

import { NextResponse } from "next/server";
import { requireAccess } from "@/lib/access";

export function proxy(request: Request) {
  return requireAccess(request) ?? NextResponse.next();
}

export const config = { matcher: ["/", "/api/:path*"] };

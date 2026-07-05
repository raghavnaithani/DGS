import { createServerClient } from "@supabase/ssr";
import { NextRequest, NextResponse } from "next/server";

/**
 * Next.js App Router middleware.
 *
 * Responsibilities:
 * 1. Create a Supabase server client using the request cookies.
 * 2. Refresh the auth session so tokens stay valid between pages.
 * 3. Redirect unauthenticated users away from protected routes to /login.
 * 4. Pass through all public routes (login, signup, share, api, static assets).
 */

/** Routes that require a valid Supabase session. */
const PROTECTED_PREFIXES = ["/dashboard", "/graph", "/onboarding", "/pricing"];

/** Routes always accessible without auth. */
const PUBLIC_PREFIXES = ["/login", "/signup", "/auth", "/share", "/api", "/_next", "/favicon"];

function isProtected(pathname: string): boolean {
  return PROTECTED_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

function isPublic(pathname: string): boolean {
  return PUBLIC_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Always pass through public paths without touching auth
  if (isPublic(pathname)) {
    return NextResponse.next();
  }

  // Create a mutable response so we can set cookie headers on it
  let response = NextResponse.next({
    request: { headers: request.headers },
  });

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value)
          );
          response = NextResponse.next({
            request,
          });
          cookiesToSet.forEach(({ name, value, options }) =>
            response.cookies.set(name, value, options)
          );
        },
      },
    }
  );

  // Refresh the session — this is a no-op if the session is still valid.
  // It also refreshes the access token cookie if it has expired.
  const {
    data: { session },
  } = await supabase.auth.getSession();

  // Redirect unauthenticated users trying to access protected routes
  if (isProtected(pathname) && !session) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("redirect", pathname);
    return NextResponse.redirect(loginUrl);
  }

  return response;
}

export const config = {
  /**
   * Run middleware on all paths EXCEPT:
   * - _next/static (static files)
   * - _next/image (image optimisation)
   * - favicon.ico
   */
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};

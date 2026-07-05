import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";

/**
 * OAuth callback handler.
 *
 * After Google OAuth, Supabase redirects here with a `code` query parameter.
 * We exchange it for a session, then redirect the user to:
 * - /onboarding if their profile is not yet complete
 * - the original redirect target or /dashboard otherwise
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const code = searchParams.get("code");
  const redirectTo = searchParams.get("redirect") ?? "/dashboard";

  if (!code) {
    return NextResponse.redirect(new URL("/login?error=missing_code", request.url));
  }

  const cookieStore = await cookies();

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value, options }) => {
            cookieStore.set(name, value, options);
          });
        },
      },
    }
  );

  const { error } = await supabase.auth.exchangeCodeForSession(code);

  if (error) {
    console.error("OAuth callback error:", error.message);
    return NextResponse.redirect(
      new URL(`/login?error=${encodeURIComponent(error.message)}`, request.url)
    );
  }

  // Check if this user needs to complete onboarding
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (session) {
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"}/v1/profile`,
        { headers: { Authorization: `Bearer ${session.access_token}` } }
      );
      if (res.status === 404) {
        return NextResponse.redirect(new URL("/onboarding", request.url));
      }
    } catch {
      // Profile check failed — go to dashboard anyway
    }
  }

  return NextResponse.redirect(new URL(redirectTo, request.url));
}

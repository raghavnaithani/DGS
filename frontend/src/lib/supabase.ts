import { createBrowserClient } from "@supabase/ssr";

/**
 * Singleton Supabase browser client.
 * Uses environment variables set in frontend/.env.local
 * Import this wherever you need Supabase auth in client components.
 */
export const supabase = createBrowserClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
);

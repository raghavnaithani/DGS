// ---------------------------------------------------------------------------
// Re-export all v0.1 API types (preserved from api.ts)
// ---------------------------------------------------------------------------

export type { JobStatus, GraphRisk, GraphNode, GraphEdge, GraphData } from "./api";

// ---------------------------------------------------------------------------
// v0.2 types
// ---------------------------------------------------------------------------

export type ExpertiseLevel = "beginner" | "intermediate" | "expert";
export type SubscriptionTier = "free" | "pro";

/** Matches the UserProfile Pydantic model returned by GET/POST /v1/profile */
export interface UserProfile {
  id: string;
  email: string;
  display_name: string | null;
  expertise_level: ExpertiseLevel;
  risk_tolerance: number; // 1-10
  values: string[];
  life_situation: string;
  decision_patterns: Record<string, unknown>;
  onboarding_complete: boolean;
  subscription_tier: SubscriptionTier;
  stripe_customer_id: string | null;
  graphs_this_month: number;
}

/** Lightweight session object returned by GET /v1/sessions */
export interface SessionSummary {
  id: string;
  title: string;
  domain: string;
  horizon_months: number;
  node_count: number;
  created_at: string;
  updated_at: string;
}

/** Returned by GET /v1/account/usage */
export interface UsageInfo {
  graphs_this_month: number;
  graphs_limit: number;
  subscription_tier: SubscriptionTier;
}

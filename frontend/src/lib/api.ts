export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type JobStatus = "queued" | "running" | "completed" | "failed";

export type GraphRisk = {
  id?: string;
  description?: string;
  severity?: string;
  likelihood?: string;
  mitigation?: string;
  mitigations?: string[];
};

export type GraphNode = {
  id: string;
  title: string;
  summary: string;
  description: string;
  time_step?: number;
  created_by_engine?: string;
  alternatives: unknown[];
  risks: GraphRisk[];
  source_citations: unknown[];
  confidence_score: number;
  speculative: boolean;
  created_at?: string;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  action_description?: string;
};

export type GraphData = {
  session_id: string;
  public_id?: string;
  title?: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export function buildApiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

/** Retrieve the Supabase access token if a session exists. Returns null otherwise. */
async function getAccessToken(): Promise<string | null> {
  try {
    // Dynamic import so this module stays usable in non-browser contexts too
    const { supabase } = await import("./supabase");
    const { data } = await supabase.auth.getSession();
    return data.session?.access_token ?? null;
  } catch {
    return null;
  }
}

export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const accessToken = await getAccessToken();
  const authHeaders: Record<string, string> = accessToken
    ? { Authorization: `Bearer ${accessToken}` }
    : {};

  const response = await fetch(buildApiUrl(path), {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders,
      ...(init?.headers ?? {}),
    },
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const detail = typeof payload === "string" ? payload : payload?.detail ?? payload?.message;
    throw new Error(detail || `Request failed (${response.status})`);
  }

  return payload as T;
}


function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function readGraphContainer(payload: unknown): Record<string, unknown> {
  if (!payload || typeof payload !== "object") {
    return {};
  }

  const container = payload as Record<string, unknown>;
  const nested = container.graph ?? container.data ?? container.result;
  if (nested && typeof nested === "object") {
    return nested as Record<string, unknown>;
  }

  return container;
}

export function normalizeGraphData(payload: unknown): GraphData {
  const container = readGraphContainer(payload);
  const nodes = asArray(container.nodes).map((item, index) => {
    const node = item && typeof item === "object" ? (item as Record<string, unknown>) : {};
    return {
      id: String(node.id ?? `node-${index + 1}`),
      title: String(node.title ?? "Untitled node"),
      summary: String(node.summary ?? ""),
      description: String(node.description ?? node.summary ?? ""),
      time_step: typeof node.time_step === "number" ? node.time_step : undefined,
      created_by_engine: node.created_by_engine ? String(node.created_by_engine) : undefined,
      alternatives: asArray(node.alternatives),
      risks: asArray(node.risks) as GraphRisk[],
      source_citations: asArray(node.source_citations),
      confidence_score: typeof node.confidence_score === "number" ? node.confidence_score : Number(node.confidence_score ?? 0),
      speculative: Boolean(node.speculative),
      created_at: node.created_at ? String(node.created_at) : undefined,
    } satisfies GraphNode;
  });

  const edges = asArray(container.edges).map((item, index) => {
    const edge = item && typeof item === "object" ? (item as Record<string, unknown>) : {};
    return {
      id: String(edge.id ?? `${String(edge.source ?? edge.from ?? "source")}-${String(edge.target ?? edge.to ?? "target")}-${index + 1}`),
      source: String(edge.source ?? edge.from ?? ""),
      target: String(edge.target ?? edge.to ?? ""),
      action_description: edge.action_description ? String(edge.action_description) : undefined,
    } satisfies GraphEdge;
  });

  return {
    session_id: String(container.session_id ?? container.sessionId ?? container.id ?? ""),
    public_id: container.public_id ? String(container.public_id) : undefined,
    title: container.title ? String(container.title) : undefined,
    nodes,
    edges,
  };
}

export function formatPublicGraphUrl(publicId: string): string {
  if (typeof window === "undefined") {
    return `/share/${publicId}`;
  }

  return `${window.location.origin}/share/${publicId}`;
}

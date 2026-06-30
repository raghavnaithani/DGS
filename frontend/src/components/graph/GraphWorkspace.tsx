"use client";

import { useCallback, useEffect, useState } from "react";

import { apiJson, normalizeGraphData, type GraphData, type JobStatus } from "@/lib/api";
import { ShareButton } from "@/components/share/ShareButton";
import { GraphCanvas } from "./GraphCanvas";

type GraphWorkspaceProps = {
  sessionId?: string;
  publicId?: string;
  readOnly?: boolean;
};

type JobResponse = {
  id: string;
  status: JobStatus;
  progress: number;
  result: { session_id?: string } | null;
  error_message?: string | null;
};

async function fetchGraph(sessionId: string): Promise<GraphData> {
  const payload = await apiJson<unknown>(`/v1/graph/${sessionId}`);
  return normalizeGraphData(payload);
}

async function fetchSharedGraph(publicId: string): Promise<GraphData> {
  const candidatePaths = [
    `/v1/graph/share/${publicId}`,
    `/v1/share/${publicId}`,
    `/v1/graph/public/${publicId}`,
  ];

  let lastError: unknown = null;
  for (const path of candidatePaths) {
    try {
      const payload = await apiJson<unknown>(path);
      return normalizeGraphData(payload);
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError instanceof Error ? lastError : new Error("Failed to load shared graph.");
}

export function GraphWorkspace({ sessionId, publicId, readOnly = false }: GraphWorkspaceProps) {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshTick, setRefreshTick] = useState(0);

  const loadGraph = useCallback(async () => {
    const identifier = sessionId ?? publicId;
    if (!identifier) {
      setGraph(null);
      setError("No graph identifier provided.");
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const nextGraph = publicId ? await fetchSharedGraph(publicId) : await fetchGraph(sessionId as string);
      setGraph(nextGraph);
    } catch (loadError) {
      setGraph(null);
      setError(loadError instanceof Error ? loadError.message : "Failed to load graph.");
    } finally {
      setLoading(false);
    }
  }, [publicId, sessionId]);

  useEffect(() => {
    void loadGraph();
  }, [loadGraph, refreshTick]);

  const pollJob = useCallback(async (jobId: string) => {
    for (let attempt = 0; attempt < 90; attempt += 1) {
      const job = await apiJson<JobResponse>(`/v1/jobs/${jobId}`);
      if (job.status === "failed") {
        throw new Error(job.error_message || "Branch job failed");
      }
      if (job.status === "completed") {
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    throw new Error("Timed out waiting for branch job");
  }, []);

  const handleBranch = useCallback(
    async (parentNodeId: string, actionDescription: string) => {
      if (!sessionId) {
        throw new Error("Branching requires a session id");
      }

      const response = await apiJson<{ job_id: string }>("/v1/simulate/branch", {
        method: "POST",
        body: JSON.stringify({
          session_id: sessionId,
          parent_node_id: parentNodeId,
          action_description: actionDescription,
        }),
      });
      await pollJob(response.job_id);
      setRefreshTick((value) => value + 1);
    },
    [pollJob, sessionId],
  );

  if (loading) {
    return <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">Loading graph…</div>;
  }

  if (error) {
    return <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error}</div>;
  }

  if (!graph) {
    return <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">No graph data available.</div>;
  }

  return (
    <div className="space-y-4">
      {!readOnly && sessionId ? <ShareButton sessionId={sessionId} /> : null}
      <GraphCanvas graph={graph} readOnly={readOnly} onBranch={readOnly ? undefined : handleBranch} />
    </div>
  );
}

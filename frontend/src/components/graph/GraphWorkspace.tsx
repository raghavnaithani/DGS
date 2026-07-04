"use client";

import { useCallback, useEffect, useState } from "react";

import { apiJson, normalizeGraphData, type GraphData, type JobStatus } from "@/lib/api";
import { ShareButton } from "@/components/share/ShareButton";
import { GraphCanvas } from "./GraphCanvas";

type GraphWorkspaceProps = {
  sessionId?: string;
  publicId?: string;
  jobId?: string;
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

export function GraphWorkspace({ sessionId, publicId, jobId, readOnly = false }: GraphWorkspaceProps) {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(!jobId); // If jobId exists, don't block on loading state
  const [error, setError] = useState<string | null>(null);
  const [refreshTick, setRefreshTick] = useState(0);
  const [jobProgress, setJobProgress] = useState<{ status: JobStatus; progress: number; message?: string } | null>(null);

  const loadGraph = useCallback(async () => {
    const identifier = sessionId ?? publicId;
    if (!identifier) {
      if (!graph) setGraph(null);
      setError("No graph identifier provided.");
      setLoading(false);
      return;
    }

    try {
      const nextGraph = publicId ? await fetchSharedGraph(publicId) : await fetchGraph(sessionId as string);
      setGraph(nextGraph);
      setError(null);
    } catch (loadError) {
      if (!graph) {
        setGraph(null);
        setError(loadError instanceof Error ? loadError.message : "Failed to load graph.");
      }
    } finally {
      setLoading(false);
    }
  }, [publicId, sessionId, graph]);

  useEffect(() => {
    void loadGraph();
  }, [loadGraph, refreshTick]);

  useEffect(() => {
    if (!jobId) return;
    let isActive = true;

    const pollInitialJob = async () => {
      try {
        for (let attempt = 0; attempt < 180; attempt += 1) { // Up to 6 minutes
          if (!isActive) break;
          const job = await apiJson<JobResponse>(`/v1/jobs/${jobId}`);
          
          if (isActive) {
            setJobProgress({ status: job.status, progress: job.progress, message: job.error_message || undefined });
          }

          if (job.status === "failed") {
            if (isActive) setError(job.error_message || "Simulation job failed");
            break;
          }
          if (job.status === "completed") {
            if (isActive) {
              setJobProgress(null);
              setRefreshTick(t => t + 1);
            }
            break;
          }
          
          if (isActive && attempt % 2 === 0) {
            setRefreshTick(t => t + 1); // Refresh graph every 4 seconds to show new nodes
          }
          await new Promise((resolve) => setTimeout(resolve, 2000));
        }
      } catch (err) {
        if (isActive) setError(err instanceof Error ? err.message : "Failed to poll job");
      }
    };

    void pollInitialJob();
    return () => { isActive = false; };
  }, [jobId]);

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

  if (loading && !graph) {
    return <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">Loading graph…</div>;
  }

  if (error && !jobProgress) {
    return <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error}</div>;
  }

  if (!graph) {
    if (jobProgress) {
      return (
        <div className="space-y-4">
          <div className="rounded-xl border border-blue-200 bg-blue-50 p-4 text-sm text-blue-700 flex items-center justify-between">
            <span>Generating Decision Graph...</span>
            <div className="flex items-center gap-4">
              <div className="w-32 bg-blue-200 rounded-full h-2.5">
                <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${jobProgress.progress}%` }}></div>
              </div>
              <span className="font-mono text-xs">{jobProgress.progress}%</span>
            </div>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600 text-center animate-pulse">
            Laying out the root decision node...
          </div>
        </div>
      );
    }
    return <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">No graph data available.</div>;
  }

  return (
    <div className="space-y-4">
      {jobProgress && (
        <div className="rounded-xl border border-blue-200 bg-blue-50 p-4 text-sm text-blue-700 flex items-center justify-between">
          <span>Generating Decision Graph...</span>
          <div className="flex items-center gap-4">
            <div className="w-32 bg-blue-200 rounded-full h-2.5">
              <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${jobProgress.progress}%` }}></div>
            </div>
            <span className="font-mono text-xs">{jobProgress.progress}%</span>
          </div>
        </div>
      )}
      {!readOnly && sessionId ? <ShareButton sessionId={sessionId} /> : null}
      {graph.nodes.length > 0 ? (
        <GraphCanvas graph={graph} readOnly={readOnly} onBranch={readOnly ? undefined : handleBranch} />
      ) : (
        <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600 text-center animate-pulse">
          Laying out the root decision node...
        </div>
      )}
    </div>
  );
}

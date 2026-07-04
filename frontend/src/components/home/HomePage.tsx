"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { apiJson, type JobStatus } from "@/lib/api";
import { IntakeWizard, type UserIntent } from "@/components/intake/IntakeWizard";

type IntentReadyOptions = {
  disableScraping: boolean;
};

type StartSimulationResponse = {
  job_id: string;
  status: "queued";
};

type JobResponse = {
  id: string;
  status: JobStatus;
  progress: number;
  result: {
    session_id?: string;
  } | null;
  error_message?: string | null;
};



export function HomePage() {
  const router = useRouter();
  const activeIntentId = useRef<string | null>(null);
  const [building, setBuilding] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const startSimulation = async (intent: UserIntent, options: IntentReadyOptions) => {
    if (activeIntentId.current === intent.id) {
      return;
    }

    activeIntentId.current = intent.id;
    setBuilding(true);
    setError(null);
    setStatusMessage("Starting simulation…");

    try {
      const response = await apiJson<StartSimulationResponse>("/v1/simulate/start", {
        method: "POST",
        body: JSON.stringify({
          user_intent_id: intent.id,
          disable_scraping: options.disableScraping,
        }),
      });
      router.push(`/graph/${intent.id}?job_id=${response.job_id}`);
    } catch (startError) {
      activeIntentId.current = null;
      setBuilding(false);
      setError(startError instanceof Error ? startError.message : "Failed to start simulation.");
      setStatusMessage(null);
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 px-4 py-8 text-slate-900 sm:px-6 lg:px-10">
      <div className="mx-auto flex max-w-5xl flex-col gap-6">
        <section className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Decision Graph Simulator</p>
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Phase 6 Frontend</h1>
          <p className="max-w-3xl text-sm leading-6 text-slate-600 sm:text-base">
            Enter a decision prompt, complete intake, and the graph will build automatically.
          </p>
        </section>

        <IntakeWizard onIntentReady={startSimulation} />

        {building || statusMessage ? (
          <section className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700">
            {statusMessage ?? "Building graph…"}
          </section>
        ) : null}

        {error ? <section className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</section> : null}
      </div>
    </main>
  );
}

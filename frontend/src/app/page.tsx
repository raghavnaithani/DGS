import { IntakeWizard } from "@/components/intake/IntakeWizard";
import { GraphCanvas } from "@/components/graph/GraphCanvas";

export default function Page() {
  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_#f8fafc,_#e2e8f0)] px-4 py-8 text-slate-900 sm:px-6 lg:px-10">
      <div className="mx-auto max-w-5xl space-y-8">
        <section className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Decision Graph Simulator</p>
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">Phase 2: Smart Intake & Clarification</h1>
          <p className="max-w-3xl text-sm leading-6 text-slate-600 sm:text-base">
            Start with a raw prompt, answer one question at a time, and generate a structured UserIntent. Use the mock flow while building, or switch to Groq-backed intake when you are ready.
          </p>
        </section>

        <IntakeWizard />

        <section className="grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
          <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold">Graph Canvas Placeholder</h2>
            <p className="mt-2 text-sm text-slate-600">The graph surface remains available for the next phase.</p>
            <div className="mt-6">
              <GraphCanvas />
            </div>
          </div>

          <aside className="rounded-3xl border border-slate-200 bg-slate-900 p-6 text-slate-50 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Mode</p>
            <h2 className="mt-2 text-xl font-semibold">Development-ready intake</h2>
            <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-300">
              <li>Mock endpoints let frontend work happen without consuming Groq quota.</li>
              <li>The real backend route validates prompt length, answers, and the final `UserIntent` schema.</li>
              <li>Both paths return the same structure, so the UI does not need to change later.</li>
            </ul>
          </aside>
        </section>
      </div>
    </main>
  );
}

import { GraphCanvas } from "@/components/graph/GraphCanvas";

export default function Page() {
  return (
    <main className="min-h-screen p-8">
      <h1 className="text-3xl font-semibold">Decision Graph Simulator</h1>
      <p className="mt-2 text-sm text-slate-600">Phase 0 scaffold</p>
      <div className="mt-8">
        <GraphCanvas />
      </div>
    </main>
  );
}

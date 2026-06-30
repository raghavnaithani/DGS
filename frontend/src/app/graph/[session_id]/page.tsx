import { GraphWorkspace } from "@/components/graph/GraphWorkspace";

type GraphPageProps = {
  params: Promise<{
    session_id: string;
  }>;
};

export default async function Page({ params }: GraphPageProps) {
  const { session_id } = await params;

  return (
    <main className="min-h-screen bg-slate-50 px-4 py-6 text-slate-900 sm:px-6 lg:px-10">
      <div className="mx-auto max-w-7xl space-y-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Decision Graph</p>
          <h1 className="text-2xl font-semibold">Session {session_id}</h1>
        </div>
        <GraphWorkspace sessionId={session_id} />
      </div>
    </main>
  );
}
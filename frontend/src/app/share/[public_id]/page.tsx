import { GraphWorkspace } from "@/components/graph/GraphWorkspace";

type SharePageProps = {
  params: Promise<{
    public_id: string;
  }>;
};

export default async function SharePage({ params }: SharePageProps) {
  const { public_id } = await params;

  return (
    <main className="min-h-screen bg-slate-50 px-4 py-6 text-slate-900 sm:px-6 lg:px-10">
      <div className="mx-auto max-w-7xl space-y-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Shared Graph</p>
          <h1 className="text-2xl font-semibold">Public view {public_id}</h1>
        </div>
        <GraphWorkspace publicId={public_id} readOnly />
      </div>
    </main>
  );
}

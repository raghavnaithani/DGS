"use client";

import { useState } from "react";

import { apiJson, formatPublicGraphUrl } from "@/lib/api";

type ShareButtonProps = {
  sessionId: string;
};

type ShareResponse = {
  public_id?: string;
  public_url?: string;
  url?: string;
  share_url?: string;
};

export function ShareButton({ sessionId }: ShareButtonProps) {
  const [loading, setLoading] = useState(false);
  const [publicUrl, setPublicUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleShare = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiJson<ShareResponse>(`/v1/graph/${sessionId}/share`, { method: "POST" });
      const url = response.public_url || response.url || response.share_url || (response.public_id ? formatPublicGraphUrl(response.public_id) : null);
      if (!url) {
        throw new Error("Share endpoint did not return a public URL");
      }
      setPublicUrl(url);
    } catch (shareError) {
      setError(shareError instanceof Error ? shareError.message : "Failed to share graph.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-700">
      <button
        type="button"
        onClick={handleShare}
        disabled={loading}
        className="inline-flex items-center justify-center rounded-full bg-slate-900 px-4 py-2 font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loading ? "Sharing…" : "Share"}
      </button>
      {publicUrl ? <p className="mt-3 break-all text-slate-600">Public URL: {publicUrl}</p> : null}
      {error ? <p className="mt-3 text-red-600">{error}</p> : null}
    </div>
  );
}

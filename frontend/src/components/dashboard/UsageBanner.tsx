"use client";

import { useState } from "react";

interface UsageBannerProps {
  graphsThisMonth: number;
  freeLimit: number;
  subscriptionTier: string;
}

export function UsageBanner({ graphsThisMonth, freeLimit, subscriptionTier }: UsageBannerProps) {
  if (subscriptionTier === "pro") {
    return null;
  }

  if (graphsThisMonth >= freeLimit) {
    return (
      <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 mb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-red-500 font-semibold">Limit Reached</h3>
          <p className="text-red-400 text-sm mt-1">You've reached your free limit for this month.</p>
        </div>
        <button className="bg-red-500 text-white px-4 py-2 rounded-md font-medium text-sm hover:bg-red-600 transition-colors whitespace-nowrap">
          Upgrade to Pro
        </button>
      </div>
    );
  }

  if (graphsThisMonth >= freeLimit - 1) {
    return (
      <div className="bg-amber-500/10 border border-amber-500/50 rounded-lg p-4 mb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-amber-500 font-semibold">Almost at Limit</h3>
          <p className="text-amber-400 text-sm mt-1">You've used {graphsThisMonth}/{freeLimit} free graphs this month. Upgrade for unlimited.</p>
        </div>
        <button className="bg-amber-500 text-white px-4 py-2 rounded-md font-medium text-sm hover:bg-amber-600 transition-colors whitespace-nowrap">
          Upgrade to Pro
        </button>
      </div>
    );
  }

  return null;
}

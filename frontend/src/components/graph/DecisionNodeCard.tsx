"use client";

import { memo } from "react";

import { Handle, NodeProps, Position } from "@xyflow/react";

import { type GraphNode } from "@/lib/api";
import { RiskBadge } from "./RiskBadge";

type DecisionNodeData = {
  node: GraphNode;
  onSelect: (nodeId: string) => void;
};

function confidenceStyles(score: number) {
  if (score >= 0.75) {
    return "#059669";
  }
  if (score >= 0.5) {
    return "#d97706";
  }
  return "#e11d48";
}

export const DecisionNodeCard = memo(function DecisionNodeCard({ data, selected }: NodeProps) {
  const { node, onSelect } = data as DecisionNodeData;
  const confidence = Math.max(0, Math.min(1, node.confidence_score));
  const confidenceColor = confidenceStyles(confidence);

  return (
    <div
      className={`w-96 rounded-xl border bg-white p-3 shadow-sm ${node.speculative ? "border-amber-500 border-dashed" : "border-slate-300"} ${selected ? "ring-2 ring-slate-900" : ""}`}
      onClick={() => onSelect(node.id)}
      title={node.speculative ? "⚠ Speculative" : node.title}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(node.id);
        }
      }}
    >
      <Handle type="target" position={Position.Top} className="!h-3 !w-3 !border-2 !border-slate-400 !bg-white" />

      <div className="space-y-2">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="truncate text-sm font-semibold text-slate-900">{node.title}</h3>
            <p className="mt-1 whitespace-pre-wrap text-xs leading-5 text-slate-600">{node.summary || node.description || "No summary"}</p>
          </div>
          {node.speculative ? <span className="shrink-0 rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-800">⚠ Speculative</span> : null}
        </div>

        <div className="space-y-1">
          <progress
            value={Math.round(confidence * 100)}
            max={100}
            aria-label={`Confidence ${Math.round(confidence * 100)}%`}
            style={{ accentColor: confidenceColor }}
            className="h-2 w-full overflow-hidden rounded-full bg-slate-200 [&::-webkit-progress-bar]:rounded-full [&::-webkit-progress-bar]:bg-slate-200 [&::-webkit-progress-value]:rounded-full [&::-moz-progress-bar]:rounded-full"
          />
          <div className="text-[11px] text-slate-500">Confidence {Math.round(confidence * 100)}%</div>
        </div>

        {node.risks.length ? (
          <div className="flex flex-wrap gap-1">
            {node.risks.slice(0, 3).map((risk, index) => (
              <RiskBadge key={`${node.id}-risk-${risk.id ?? index}`} risk={risk} compact />
            ))}
          </div>
        ) : null}
      </div>

      <Handle type="source" position={Position.Bottom} className="!h-3 !w-3 !border-2 !border-slate-400 !bg-white" />
    </div>
  );
});

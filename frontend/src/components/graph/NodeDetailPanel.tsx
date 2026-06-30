"use client";

import { type GraphNode } from "@/lib/api";
import { RiskBadge } from "./RiskBadge";

type NodeDetailPanelProps = {
  node: GraphNode | null;
  readOnly?: boolean;
  actionDescription: string;
  onActionDescriptionChange: (value: string) => void;
  onBranch: () => void;
  branching: boolean;
};

function formatConfidence(score: number) {
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`;
}

function renderValue(value: unknown) {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return "";
}

function renderAlternative(alternative: unknown) {
  if (!alternative || typeof alternative !== "object") {
    return renderValue(alternative);
  }

  const item = alternative as Record<string, unknown>;
  const description = typeof item.description === "string" ? item.description : "";
  const actionType = typeof item.action_type === "string" ? item.action_type : "";
  const outcome = typeof item.expected_outcome_summary === "string" ? item.expected_outcome_summary : "";

  return (
    <div className="space-y-1">
      <div className="font-medium text-slate-900">{description || renderValue(alternative)}</div>
      {(actionType || outcome) && (
        <div className="text-xs leading-5 text-slate-500">
          {actionType ? <span>Type: {actionType}</span> : null}
          {actionType && outcome ? <span> · </span> : null}
          {outcome ? <span>Expected outcome: {outcome}</span> : null}
        </div>
      )}
    </div>
  );
}

export function NodeDetailPanel({ node, readOnly = false, actionDescription, onActionDescriptionChange, onBranch, branching }: NodeDetailPanelProps) {
  if (!node) {
    return (
      <aside className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">
        Select a node to view details.
      </aside>
    );
  }

  return (
    <aside className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="space-y-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Node Details</p>
          <h2 className="mt-1 text-lg font-semibold text-slate-900">{node.title}</h2>
        </div>

        <div className="space-y-2 text-sm text-slate-700">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Description</div>
            <p className="mt-1 leading-6">{node.description || "No description available."}</p>
          </div>

          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Alternatives</div>
            {node.alternatives.length ? (
              <ul className="mt-1 list-disc space-y-1 pl-5">
                {node.alternatives.map((alternative, index) => (
                  <li key={`${node.id}-alt-${index}`}>{renderAlternative(alternative)}</li>
                ))}
              </ul>
            ) : (
              <p className="mt-1 text-slate-500">No alternatives listed.</p>
            )}
          </div>

          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Risks</div>
            {node.risks.length ? (
              <div className="mt-2 space-y-2">
                {node.risks.map((risk, index) => (
                  <div key={`${node.id}-risk-detail-${risk.id ?? index}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
                    <div className="flex flex-wrap items-center gap-2">
                      <RiskBadge risk={risk} />
                      {risk.likelihood ? <span className="text-xs text-slate-500">Likelihood: {risk.likelihood}</span> : null}
                    </div>
                    {risk.mitigation ? <p className="mt-2 text-sm text-slate-600">Mitigation: {risk.mitigation}</p> : null}
                    {risk.mitigations?.length ? <p className="mt-2 text-sm text-slate-600">Mitigations: {risk.mitigations.join(", ")}</p> : null}
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-1 text-slate-500">No risks listed.</p>
            )}
          </div>

          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Source Citations</div>
            {node.source_citations.length ? (
              <ul className="mt-1 list-disc space-y-1 pl-5 text-slate-600">
                {node.source_citations.map((citation, index) => (
                  <li key={`${node.id}-citation-${index}`}>{renderValue(citation)}</li>
                ))}
              </ul>
            ) : (
              <p className="mt-1 text-slate-500">No citations listed.</p>
            )}
          </div>

          <div className="flex flex-wrap gap-4">
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Confidence</div>
              <div className="mt-1 text-sm text-slate-900">{formatConfidence(node.confidence_score)}</div>
            </div>
            {node.time_step !== undefined ? (
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Time Step</div>
                <div className="mt-1 text-sm text-slate-900">{node.time_step}</div>
              </div>
            ) : null}
          </div>
        </div>

        {!readOnly ? (
          <div className="border-t border-slate-200 pt-3">
            <label className="block text-sm font-medium text-slate-700" htmlFor="branch-action-description">
              Branch action description
            </label>
            <textarea
              id="branch-action-description"
              value={actionDescription}
              onChange={(event) => onActionDescriptionChange(event.target.value)}
              rows={4}
              placeholder="Describe the branch you want to simulate"
              className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 outline-none focus:border-slate-900"
            />
            <button
              type="button"
              onClick={onBranch}
              disabled={branching}
              className="mt-3 inline-flex items-center justify-center rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
            >
              {branching ? "Branching…" : "Branch from here"}
            </button>
          </div>
        ) : null}
      </div>
    </aside>
  );
}

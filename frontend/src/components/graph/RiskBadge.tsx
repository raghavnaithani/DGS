import { type GraphRisk } from "@/lib/api";

type RiskBadgeProps = {
  risk: GraphRisk;
  compact?: boolean;
};

function severityStyles(severity?: string) {
  const value = severity?.toLowerCase() ?? "";
  if (value.includes("critical")) {
    return { label: "Critical", className: "border-red-900 bg-red-950 text-white" };
  }
  if (value.includes("high")) {
    return { label: "High", className: "border-red-300 bg-red-100 text-red-800" };
  }
  if (value.includes("medium")) {
    return { label: "Medium", className: "border-amber-300 bg-amber-100 text-amber-800" };
  }
  return { label: "Low", className: "border-emerald-300 bg-emerald-100 text-emerald-800" };
}

export function RiskBadge({ risk, compact = false }: RiskBadgeProps) {
  const severity = severityStyles(risk.severity);
  const label = risk.description?.trim() || "Risk";

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${severity.className} ${compact ? "max-w-full" : ""}`}
      title={label}
    >
      <span>{severity.label}</span>
      <span className="truncate">{label}</span>
    </span>
  );
}

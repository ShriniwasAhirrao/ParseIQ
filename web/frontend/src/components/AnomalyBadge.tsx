import { ANOMALY_SEVERITY, type Severity } from "../lib/types";
import clsx from "clsx";

const SEVERITY_STYLES: Record<Severity, string> = {
  critical: "bg-red-500/15 text-red-400 border-red-500/30",
  high: "bg-orange-500/15 text-orange-400 border-orange-500/30",
  medium: "bg-yellow-500/15 text-yellow-400 border-yellow-500/30",
  low: "bg-blue-500/15 text-blue-400 border-blue-500/30",
};

interface Props {
  anomaly: string;
  className?: string;
}

export default function AnomalyBadge({ anomaly, className }: Props) {
  const severity = ANOMALY_SEVERITY[anomaly] || "medium";
  const display = anomaly.replace(/_/g, " ");

  return (
    <span
      className={clsx(
        "inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium border",
        "font-mono tracking-tight",
        SEVERITY_STYLES[severity],
        className
      )}
    >
      <span className="sr-only">{severity} severity:</span>
      {display}
    </span>
  );
}

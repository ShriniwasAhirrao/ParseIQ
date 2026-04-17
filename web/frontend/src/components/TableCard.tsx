import { Table2, AlertTriangle, Rows3, FolderTree, ChevronRight } from "lucide-react";
import { getScoreColor } from "../lib/types";
import type { TableSummary } from "../lib/types";
import clsx from "clsx";

interface Props {
  table: TableSummary;
  onClick: () => void;
}

export default function TableCard({ table, onClick }: Props) {
  const color = getScoreColor(table.quality_score);
  const childCount = table.children?.length ?? 0;

  return (
    <button
      onClick={onClick}
      className={clsx(
        "w-full text-left rounded-xl border border-surface-300/10 bg-surface-900/60",
        "p-5 transition-all duration-200",
        "hover:border-brand-500/30 hover:bg-surface-900/90 hover:scale-[1.01]",
        "group cursor-pointer"
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-brand-500/10 flex items-center justify-center">
            <Table2 size={16} className="text-brand-400" />
          </div>
          <div>
            <h3 className="text-surface-100 font-semibold text-sm group-hover:text-brand-300 transition-colors">
              {table.name}
            </h3>
            <p className="text-surface-300 text-xs mt-0.5">
              {table.attribute_count} columns
            </p>
          </div>
        </div>

        {/* Score circle */}
        <div
          className="w-10 h-10 rounded-full border-2 flex items-center justify-center text-xs font-bold"
          style={{ borderColor: color, color }}
        >
          {Math.round(table.quality_score)}
        </div>
      </div>

      {/* Stats row */}
      <div className="flex items-center gap-4 text-xs text-surface-300 flex-wrap">
        <span className="flex items-center gap-1.5">
          <Rows3 size={12} />
          {table.record_count.toLocaleString()} rows
        </span>
        {table.anomaly_count > 0 && (
          <span className="flex items-center gap-1.5 text-orange-400">
            <AlertTriangle size={12} />
            {table.anomaly_count} anomalies
          </span>
        )}
        {childCount > 0 && (
          <span className="flex items-center gap-1.5 text-brand-300">
            <FolderTree size={12} />
            {childCount} nested
            <ChevronRight size={11} className="opacity-70" />
          </span>
        )}
      </div>

      {/* Quality bar */}
      <div className="mt-3 h-1 rounded-full bg-surface-800 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${table.quality_score}%`,
            background: color,
            boxShadow: `0 0 8px ${color}40`,
          }}
        />
      </div>
    </button>
  );
}

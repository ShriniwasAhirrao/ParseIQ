import { AlertTriangle } from "lucide-react";
import clsx from "clsx";

interface Props {
  data: Record<string, unknown>[];
  maxRows?: number;
  flaggedColumns?: string[];
  anomaliesByColumn?: Record<string, string[]>;
}

export default function DataPreviewTable({
  data,
  maxRows = 50,
  flaggedColumns = [],
  anomaliesByColumn = {},
}: Props) {
  if (!data.length) {
    return (
      <p className="text-surface-300 text-sm py-8 text-center">
        No data preview available
      </p>
    );
  }

  const columns = Object.keys(data[0]);
  const rows = data.slice(0, maxRows);
  const flaggedSet = new Set(flaggedColumns);

  return (
    <>
      {flaggedColumns.length > 0 && (
        <div className="mb-3 flex items-start gap-2 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200">
          <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
          <span>
            <strong>{flaggedColumns.length}</strong> column
            {flaggedColumns.length === 1 ? "" : "s"} flagged — highlighted below:{" "}
            <span className="font-mono">{flaggedColumns.join(", ")}</span>
          </span>
        </div>
      )}

      <div className="overflow-auto rounded-xl border border-surface-300/10">
        <table className="w-full text-sm">
          <caption className="sr-only">Data preview showing first {rows.length} rows</caption>
          <thead>
            <tr className="bg-surface-900/80 border-b border-surface-300/10">
              {columns.map((col) => {
                const flagged = flaggedSet.has(col);
                const tip = flagged
                  ? (anomaliesByColumn[col] || []).join(", ")
                  : undefined;
                return (
                  <th
                    key={col}
                    title={tip}
                    className={clsx(
                      "px-4 py-2.5 text-left text-xs font-semibold font-mono whitespace-nowrap",
                      flagged
                        ? "text-amber-200 bg-amber-500/10 border-b-2 border-amber-500/40"
                        : "text-surface-200"
                    )}
                  >
                    <span className="inline-flex items-center gap-1.5">
                      {flagged && <AlertTriangle size={11} className="text-amber-400" />}
                      {col}
                    </span>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={i}
                className="border-b border-surface-300/5 transition-colors hover:bg-surface-800/40"
              >
                {columns.map((col) => {
                  const flagged = flaggedSet.has(col);
                  const value = row[col];
                  const isNull = value === null || value === undefined;
                  return (
                    <td
                      key={col}
                      className={clsx(
                        "px-4 py-2 whitespace-nowrap max-w-[300px] truncate",
                        flagged
                          ? "bg-amber-500/[0.06] text-amber-100 border-l border-amber-500/20"
                          : "text-surface-300",
                        flagged && isNull && "bg-red-500/10"
                      )}
                    >
                      {isNull ? (
                        <span className={clsx(
                          "italic",
                          flagged ? "text-red-300/80" : "text-surface-300/30"
                        )}>
                          null
                        </span>
                      ) : (
                        String(value)
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        {data.length > maxRows && (
          <div className="px-4 py-2 text-xs text-surface-300 text-center bg-surface-900/40">
            Showing {maxRows} of {data.length} rows
          </div>
        )}
      </div>
    </>
  );
}

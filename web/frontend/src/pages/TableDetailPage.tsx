import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  ArrowLeft,
  ChevronRight,
  Columns3,
  FolderTree,
  Home,
  Rows3,
  Shield,
} from "lucide-react";
import ScoreGauge from "../components/ScoreGauge";
import AnomalyBadge from "../components/AnomalyBadge";
import DataPreviewTable from "../components/DataPreviewTable";
import TableCard from "../components/TableCard";
import { getTableDetail } from "../lib/api";
import { getScoreColor } from "../lib/types";
import type { TableDetail } from "../lib/types";

export default function TableDetailPage() {
  const { jobId, tableName } = useParams<{ jobId: string; tableName: string }>();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<TableDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"columns" | "data" | "nested">("columns");

  useEffect(() => { document.title = `ParseIQ — ${tableName ?? "Table"}`; }, [tableName]);

  useEffect(() => {
    if (!jobId || !tableName) return;
    setLoading(true);
    setDetail(null);
    setError(null);
    getTableDetail(jobId, tableName)
      .then((d) => {
        setDetail(d);
        if (d.children && d.children.length > 0 && (d.data_preview?.length ?? 0) === 0) {
          setTab("nested");
        }
      })
      .catch((err) => {
        setError(err?.response?.data?.error || err?.message || "Failed to load table");
      })
      .finally(() => setLoading(false));
  }, [jobId, tableName]);

  if (loading) {
    return (
      <div className="noise-bg min-h-[calc(100vh-56px)]">
        <div className="max-w-7xl mx-auto px-6 py-10">
          <div className="h-4 w-40 bg-surface-800/60 rounded animate-pulse mb-5" />
          <div className="h-8 w-56 bg-surface-800 rounded-lg animate-pulse mb-3" />
          <div className="h-4 w-72 bg-surface-800/60 rounded animate-pulse mb-8" />
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="rounded-xl border border-surface-300/10 bg-surface-900/60 p-4 h-24 animate-pulse" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="min-h-[calc(100vh-56px)] flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 mb-2">{error || "Table not found"}</p>
          <button
            onClick={() => navigate(-1)}
            className="px-4 py-2 rounded-lg bg-surface-800 text-surface-200 text-sm hover:bg-surface-800/80 transition-colors cursor-pointer"
          >
            Go back
          </button>
        </div>
      </div>
    );
  }

  const attrs = Object.entries(detail.attributes);
  const anomalyCount = Object.values(detail.anomalies).reduce(
    (s, flags) => s + flags.length,
    0
  );
  const flaggedColumns = Object.keys(detail.anomalies).filter(
    (c) => (detail.anomalies[c]?.length ?? 0) > 0
  );
  const children = detail.children ?? [];

  return (
    <div className="noise-bg min-h-[calc(100vh-56px)]">
      <div className="max-w-7xl mx-auto px-6 py-10">
        {/* Breadcrumbs */}
        <nav className="flex items-center gap-1.5 text-xs text-surface-300 mb-5 flex-wrap">
          <Link
            to={`/results/${jobId}`}
            className="flex items-center gap-1 hover:text-surface-100 transition-colors"
          >
            <Home size={11} /> Dashboard
          </Link>
          {detail.parent && (
            <>
              <ChevronRight size={12} className="opacity-50" />
              <Link
                to={`/results/${jobId}/table/${detail.parent}`}
                className="hover:text-surface-100 transition-colors font-mono"
              >
                {detail.parent}
              </Link>
            </>
          )}
          <ChevronRight size={12} className="opacity-50" />
          <span className="text-surface-100 font-mono font-semibold">
            {detail.name}
          </span>
        </nav>

        <button
          onClick={() => navigate(-1)}
          className="flex items-center gap-2 text-surface-300 text-sm hover:text-surface-100 transition-colors mb-6 cursor-pointer"
        >
          <ArrowLeft size={14} />
          Back
        </button>

        {/* Header row */}
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6 mb-8 animate-fade-up">
          <div>
            <h1 className="text-3xl font-bold text-surface-100 font-[family-name:var(--font-display)] tracking-tight">
              {detail.name}
            </h1>
            <div className="flex items-center gap-5 mt-3 text-sm text-surface-300 flex-wrap">
              <span className="flex items-center gap-1.5">
                <Rows3 size={14} /> {detail.record_count.toLocaleString()} rows
              </span>
              <span className="flex items-center gap-1.5">
                <Columns3 size={14} /> {attrs.length} columns
              </span>
              <span className="flex items-center gap-1.5">
                <Shield size={14} /> {anomalyCount} anomalies
              </span>
              {children.length > 0 && (
                <span className="flex items-center gap-1.5 text-brand-300">
                  <FolderTree size={14} /> {children.length} nested
                </span>
              )}
            </div>
          </div>

          <ScoreGauge score={detail.quality_score} size={120} />
        </div>

        {/* Anomalies row */}
        {anomalyCount > 0 && (
          <div className="mb-8 animate-fade-up animate-fade-up-delay-1">
            <h2 className="text-xs font-semibold text-surface-200 mb-3 uppercase tracking-wider">
              Anomalies
            </h2>
            <div className="flex flex-wrap gap-2">
              {Object.entries(detail.anomalies).map(([col, flags]) =>
                flags.map((flag) => (
                  <div key={`${col}-${flag}`} className="flex items-center gap-1.5">
                    <span className="text-xs text-surface-300 font-mono">{col}:</span>
                    <AnomalyBadge anomaly={flag} />
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Tab switcher */}
        <div role="tablist" className="flex gap-1 mb-6 p-1 rounded-lg bg-surface-900/60 border border-surface-300/10 w-fit animate-fade-up animate-fade-up-delay-2">
          {(["columns", "data", "nested"] as const).map((t) => {
            const disabled = t === "nested" && children.length === 0;
            const label =
              t === "columns"
                ? "Column Profiles"
                : t === "data"
                ? "Data Preview"
                : `Nested Tables (${children.length})`;
            return (
              <button
                key={t}
                role="tab"
                aria-selected={tab === t}
                onClick={() => !disabled && setTab(t)}
                disabled={disabled}
                className={
                  "px-4 py-1.5 rounded-md text-sm font-medium transition-all " +
                  (disabled
                    ? "text-surface-300/30 cursor-not-allowed"
                    : tab === t
                    ? "bg-brand-500/20 text-brand-300 cursor-pointer"
                    : "text-surface-300 hover:text-surface-100 cursor-pointer")
                }
              >
                {label}
              </button>
            );
          })}
        </div>

        {/* Tab content */}
        {tab === "columns" ? (
          <div className="space-y-3 animate-fade-up">
            {attrs.map(([name, attr]) => {
              const a = attr as Record<string, unknown>;
              const score = (a.quality_score as number) ?? 100;
              const color = getScoreColor(score);
              const colAnomalies = detail.anomalies[name] || [];

              return (
                <div
                  key={name}
                  className="rounded-xl border border-surface-300/10 bg-surface-900/60 p-4"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <span className="text-surface-100 font-semibold text-sm font-mono">
                        {name}
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-surface-800 text-surface-300">
                        {String(a.data_type || "unknown")}
                      </span>
                    </div>
                    <span className="text-sm font-bold" style={{ color }}>
                      {Math.round(score)}
                    </span>
                  </div>

                  <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-surface-300">
                    <span>
                      Present: {String(a.present_count ?? "—")} / {String(a.total_count ?? "—")}
                    </span>
                    <span>Missing: {String(a.missing_percentage ?? 0)}%</span>
                    <span>Unique: {String(a.unique_values ?? "—")}</span>
                  </div>

                  {colAnomalies.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mt-2">
                      {colAnomalies.map((f) => (
                        <AnomalyBadge key={f} anomaly={f} />
                      ))}
                    </div>
                  )}

                  <div className="mt-2 h-1 rounded-full bg-surface-800 overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${score}%`, background: color }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        ) : tab === "data" ? (
          <div className="animate-fade-up">
            <DataPreviewTable
              data={detail.data_preview}
              flaggedColumns={flaggedColumns}
              anomaliesByColumn={detail.anomalies}
            />
          </div>
        ) : (
          <div className="animate-fade-up space-y-4">
            <p className="text-xs text-surface-300">
              These tables are nested inside <span className="font-mono text-surface-100">{detail.name}</span>.
              Click any card to drill deeper.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {children.map((c) => (
                <TableCard
                  key={c.name}
                  table={c}
                  onClick={() => navigate(`/results/${jobId}/table/${encodeURIComponent(c.name)}`)}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

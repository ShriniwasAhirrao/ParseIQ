import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { CheckCircle2, Download, Table2, AlertTriangle, Layers, FolderTree, Plus } from "lucide-react";
import ScoreGauge from "../components/ScoreGauge";
import TableCard from "../components/TableCard";
import AnomalyBadge from "../components/AnomalyBadge";
import { getResults, getDownloadUrl } from "../lib/api";
import type { AnalysisResult } from "../lib/types";

export default function DashboardPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => { document.title = "ParseIQ — Quality Report"; }, []);

  useEffect(() => {
    if (!jobId) return;
    getResults(jobId)
      .then(setResult)
      .catch((err) => {
        const msg = err?.response?.data?.error || err?.message || "Failed to load results";
        setError(msg);
      })
      .finally(() => setLoading(false));
  }, [jobId]);

  if (loading) {
    return (
      <div className="noise-bg min-h-[calc(100vh-56px)]">
        <div className="max-w-7xl mx-auto px-6 py-10">
          <div className="h-8 w-48 bg-surface-800 rounded-lg animate-pulse mb-2" />
          <div className="h-4 w-64 bg-surface-800/60 rounded animate-pulse mb-10" />
          <div className="grid grid-cols-1 md:grid-cols-4 gap-5 mb-10">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="rounded-xl border border-surface-300/10 bg-surface-900/60 p-6 h-32 animate-pulse" />
            ))}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="rounded-xl border border-surface-300/10 bg-surface-900/60 p-5 h-40 animate-pulse" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="min-h-[calc(100vh-56px)] flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 mb-2">{error || "Results not found"}</p>
          <button
            onClick={() => navigate("/")}
            className="px-4 py-2 rounded-lg bg-surface-800 text-surface-200 text-sm hover:bg-surface-800/80 transition-colors cursor-pointer"
          >
            Back to upload
          </button>
        </div>
      </div>
    );
  }

  // Collect all unique anomaly types across tables
  const allAnomalies: string[] = [];
  const seen = new Set<string>();
  Object.values(result.anomalies).forEach((cols) =>
    Object.values(cols).forEach((flags) =>
      flags.forEach((f) => {
        if (!seen.has(f)) {
          seen.add(f);
          allAnomalies.push(f);
        }
      })
    )
  );

  return (
    <div className="noise-bg min-h-[calc(100vh-56px)]">
      <div className="max-w-7xl mx-auto px-6 py-10">
        {/* Header */}
        <div className="flex flex-col sm:flex-row items-start justify-between gap-4 mb-10 animate-fade-up">
          <div>
            <h1 className="text-3xl font-bold text-surface-100 font-[family-name:var(--font-display)] tracking-tight">
              Quality Report
            </h1>
            <p className="text-surface-300 text-sm mt-1">
              {result.tables.length} tables analyzed &middot; {result.total_anomalies} anomalies detected
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate("/")}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-surface-800 text-surface-200 text-sm font-medium hover:bg-surface-800/80 transition-colors cursor-pointer"
            >
              <Plus size={14} />
              New Analysis
            </button>
            <a
              href={getDownloadUrl(jobId!)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-brand-500 text-white text-sm font-medium hover:bg-brand-600 transition-colors"
            >
              <Download size={14} />
              Excel Report
            </a>
          </div>
        </div>

        {/* Top stats row */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-5 mb-10">
          {/* Score gauge — spans 1 col */}
          <div className="rounded-xl border border-surface-300/10 bg-surface-900/60 p-6 flex items-center justify-center animate-fade-up animate-fade-up-delay-1">
            <ScoreGauge score={result.overall_quality_score} label="Overall Quality" />
          </div>

          {/* Stat cards */}
          <StatCard
            icon={<Layers size={18} />}
            label="Tables"
            value={result.tables.length}
            className="animate-fade-up animate-fade-up-delay-2"
          />
          <StatCard
            icon={<Table2 size={18} />}
            label="Total Columns"
            value={result.table_summaries.reduce((s, t) => s + t.attribute_count, 0)}
            className="animate-fade-up animate-fade-up-delay-3"
          />
          <StatCard
            icon={<AlertTriangle size={18} />}
            label="Anomalies"
            value={result.total_anomalies}
            valueColor={result.total_anomalies > 0 ? "text-orange-400" : "text-green-400"}
            className="animate-fade-up animate-fade-up-delay-4"
          />
        </div>

        {/* Anomaly summary */}
        {allAnomalies.length > 0 ? (
          <div className="mb-10 animate-fade-up">
            <h2 className="text-sm font-semibold text-surface-200 mb-3 uppercase tracking-wider">
              Anomalies Found
            </h2>
            <div className="flex flex-wrap gap-2">
              {allAnomalies.map((a) => (
                <AnomalyBadge key={a} anomaly={a} />
              ))}
            </div>
          </div>
        ) : (
          <div className="mb-10 animate-fade-up p-4 rounded-xl border border-green-500/20 bg-green-500/5 text-green-400 text-sm flex items-center gap-2">
            <CheckCircle2 size={16} />
            No anomalies detected — your data looks clean!
          </div>
        )}

        {/* Table grid */}
        {(() => {
          const roots = result.table_summaries.filter((t) => !t.parent);
          const hasHierarchy = result.table_summaries.some((t) => t.parent);
          const shown = showAll ? result.table_summaries : roots;
          const hiddenChildren = result.table_summaries.length - roots.length;
          return (
            <>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-surface-200 uppercase tracking-wider flex items-center gap-2">
                  {showAll ? "All tables" : "Root tables"}
                  {hasHierarchy && (
                    <span className="inline-flex items-center gap-1 text-[10px] text-brand-300 bg-brand-500/10 border border-brand-500/20 px-1.5 py-0.5 rounded font-mono normal-case tracking-normal">
                      <FolderTree size={10} /> nested
                    </span>
                  )}
                </h2>
                {hasHierarchy && (
                  <button
                    onClick={() => setShowAll(!showAll)}
                    className="text-xs text-surface-300 hover:text-surface-100 transition-colors cursor-pointer"
                  >
                    {showAll
                      ? `Show only roots (${roots.length})`
                      : `Show all ${result.table_summaries.length} tables (${hiddenChildren} nested hidden)`}
                  </button>
                )}
              </div>
              {hasHierarchy && !showAll && (
                <p className="text-xs text-surface-300/70 mb-4">
                  Tables below contain nested child tables. Click any card to drill in.
                </p>
              )}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {shown.map((t) => (
                  <TableCard
                    key={t.name}
                    table={t}
                    onClick={() => navigate(`/results/${jobId}/table/${encodeURIComponent(t.name)}`)}
                  />
                ))}
              </div>
            </>
          );
        })()}
      </div>
    </div>
  );
}

/* ── Mini stat card ─────────────────────────────────── */

function StatCard({
  icon,
  label,
  value,
  valueColor = "text-surface-100",
  className = "",
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  valueColor?: string;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl border border-surface-300/10 bg-surface-900/60 p-5 flex flex-col gap-3 ${className}`}
    >
      <div className="w-9 h-9 rounded-lg bg-brand-500/10 flex items-center justify-center text-brand-400">
        {icon}
      </div>
      <div>
        <p className={`text-2xl font-bold ${valueColor} font-[family-name:var(--font-display)]`}>
          {value.toLocaleString()}
        </p>
        <p className="text-xs text-surface-300 mt-0.5">{label}</p>
      </div>
    </div>
  );
}

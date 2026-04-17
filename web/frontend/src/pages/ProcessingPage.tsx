import React, { useEffect, useRef } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { AlertCircle, Brain, CheckCircle2, Loader2, Sparkles } from "lucide-react";
import ProgressSteps from "../components/ProgressSteps";
import { useJobPoller } from "../hooks/useJobPoller";
import type { JobEvent } from "../lib/types";

export default function ProcessingPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const state = useJobPoller(jobId ?? null);
  const feedRef = useRef<HTMLDivElement>(null);

  useEffect(() => { document.title = "ParseIQ — Analyzing..."; }, []);

  useEffect(() => {
    if (state?.status === "completed") {
      const t = setTimeout(() => navigate(`/results/${jobId}`), 800);
      return () => clearTimeout(t);
    }
  }, [state, jobId, navigate]);

  // Auto-scroll the thought feed
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [state?.allEvents.length]);

  const failed = state?.status === "failed";
  const running = state?.status === "running" || state?.status === "pending";

  return (
    <div className="noise-bg min-h-[calc(100vh-56px)] px-6 py-10">
      <div className="w-full max-w-3xl mx-auto">
        <div className="text-center animate-fade-up">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-300 text-xs font-medium mb-4">
            {running ? (
              <Loader2 size={12} className="animate-spin" />
            ) : failed ? (
              <AlertCircle size={12} />
            ) : (
              <CheckCircle2 size={12} />
            )}
            {running ? "Working..." : failed ? "Failed" : "Complete"}
          </div>
          <h2 className="text-2xl font-bold text-surface-100 font-[family-name:var(--font-display)] mb-1">
            Analyzing your data
          </h2>
          <p className="text-surface-300 text-sm">
            ParseIQ is profiling every table, column, and record
          </p>
        </div>

        {failed ? (
          <div className="mt-10 flex flex-col items-center gap-4 animate-fade-up">
            <div className="w-14 h-14 rounded-xl bg-red-500/15 flex items-center justify-center">
              <AlertCircle size={24} className="text-red-400" />
            </div>
            <p className="text-red-400 font-medium">Analysis failed</p>
            <p className="text-surface-300 text-sm max-w-sm text-center">
              {state?.error}
            </p>
            <button
              onClick={() => navigate("/")}
              className="mt-4 px-5 py-2 rounded-lg bg-surface-800 text-surface-200 text-sm hover:bg-surface-800/80 transition-colors cursor-pointer"
            >
              Try again
            </button>
          </div>
        ) : (
          <div className="mt-10 animate-fade-up animate-fade-up-delay-1">
            <ProgressSteps
              currentStep={state?.step ?? "extract"}
              progress={state?.progress ?? 5}
              status={state?.status ?? "running"}
            />
          </div>
        )}

        {/* Thought-process feed */}
        <div className="mt-10 animate-fade-up animate-fade-up-delay-2">
          <div className="flex items-center gap-2 mb-3">
            <Brain size={14} className="text-brand-400" />
            <h3 className="text-xs font-semibold text-surface-200 uppercase tracking-wider">
              Thought process
            </h3>
            <span className="text-xs text-surface-300">
              · {state?.allEvents.length ?? 0} events
            </span>
          </div>
          <div
            ref={feedRef}
            className="rounded-xl border border-surface-300/10 bg-surface-900/70 backdrop-blur-sm font-mono text-xs max-h-72 overflow-auto"
          >
            {!state?.allEvents || state.allEvents.length === 0 ? (
              <div className="flex items-center justify-center gap-2 py-10 text-surface-300/60">
                <Loader2 size={14} className="animate-spin" />
                Warming up…
              </div>
            ) : (
              <ul className="divide-y divide-surface-300/5">
                {state.allEvents.map((e, i) => (
                  <EventRow key={i} event={e} />
                ))}
                {running && (
                  <li className="flex items-center gap-2 px-4 py-2 text-brand-300/80">
                    <Sparkles size={11} className="animate-pulse" />
                    <span className="italic">thinking…</span>
                  </li>
                )}
              </ul>
            )}
          </div>
          <p className="mt-2 text-[11px] text-surface-300/70 text-center">
            LLM calls can take 10–60s. The feed updates live so you know it's working.
          </p>
        </div>
      </div>
    </div>
  );
}

const EventRow = React.memo(function EventRow({ event }: { event: JobEvent }) {
  const t = new Date(event.ts * 1000);
  const hh = String(t.getHours()).padStart(2, "0");
  const mm = String(t.getMinutes()).padStart(2, "0");
  const ss = String(t.getSeconds()).padStart(2, "0");

  const isStage = event.level === "stage";
  const isError = event.level === "error";

  return (
    <li
      className={
        "flex items-start gap-3 px-4 py-1.5 " +
        (isStage
          ? "bg-brand-500/5 text-brand-200"
          : isError
          ? "bg-red-500/5 text-red-300"
          : "text-surface-300")
      }
    >
      <span className="text-surface-300/40 whitespace-nowrap shrink-0">
        {hh}:{mm}:{ss}
      </span>
      <span
        className={
          "whitespace-nowrap shrink-0 " +
          (isStage
            ? "text-brand-400"
            : isError
            ? "text-red-400"
            : "text-surface-300/40")
        }
      >
        {isStage ? "▶" : isError ? "✕" : "·"}
      </span>
      <span className="break-words">{event.message}</span>
    </li>
  );
});

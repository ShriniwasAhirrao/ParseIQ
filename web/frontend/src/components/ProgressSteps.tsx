import { Database, Brain, FileOutput, Loader2, CheckCircle2 } from "lucide-react";
import clsx from "clsx";

const STEPS = [
  { key: "extract", label: "Extract Metadata", icon: Database },
  { key: "enrich", label: "LLM Enrichment", icon: Brain },
  { key: "output", label: "Generate Output", icon: FileOutput },
] as const;

interface Props {
  currentStep: string | null;
  progress: number;
  status: string;
}

export default function ProgressSteps({ currentStep, progress, status }: Props) {
  const currentIdx = STEPS.findIndex((s) => s.key === currentStep);
  const done = status === "completed";

  return (
    <div className="flex flex-col items-center gap-8">
      {/* Steps */}
      <div className="flex items-center gap-3">
        {STEPS.map((step, i) => {
          const isActive = step.key === currentStep;
          const isPast = done || i < currentIdx;
          const Icon = step.icon;

          return (
            <div key={step.key} className="flex items-center gap-3">
              <div className="flex flex-col items-center gap-2">
                <div
                  className={clsx(
                    "w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-500",
                    isPast && "bg-green-500/20 text-green-400",
                    isActive && "bg-brand-500/20 text-brand-400 animate-pulse",
                    !isPast && !isActive && "bg-surface-800 text-surface-300/40"
                  )}
                >
                  {isPast ? (
                    <CheckCircle2 size={20} />
                  ) : isActive ? (
                    <Loader2 size={20} className="animate-spin" />
                  ) : (
                    <Icon size={20} />
                  )}
                </div>
                <span
                  className={clsx(
                    "text-xs font-medium",
                    isPast && "text-green-400",
                    isActive && "text-brand-300",
                    !isPast && !isActive && "text-surface-300/40"
                  )}
                >
                  {step.label}
                </span>
              </div>

              {/* Connector */}
              {i < STEPS.length - 1 && (
                <div className="w-16 h-px bg-surface-800 relative mb-6">
                  <div
                    className="absolute inset-y-0 left-0 bg-brand-500 transition-all duration-700"
                    style={{
                      width: isPast ? "100%" : isActive ? "50%" : "0%",
                    }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-md">
        <div className="flex justify-between text-xs text-surface-300 mb-2">
          <span>
            {done ? "Analysis complete" : `Analyzing...`}
          </span>
          <span>{progress}%</span>
        </div>
        <div
          className="h-2 rounded-full bg-surface-800 overflow-hidden"
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Analysis progress"
        >
          <div
            className="h-full rounded-full transition-all duration-500 ease-out"
            style={{
              width: `${progress}%`,
              background: done
                ? "var(--color-quality-excellent)"
                : "var(--color-brand-500)",
              boxShadow: `0 0 12px ${done ? "var(--color-quality-excellent)" : "var(--color-brand-500)"}40`,
            }}
          />
        </div>
      </div>
    </div>
  );
}

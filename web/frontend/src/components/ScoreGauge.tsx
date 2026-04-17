import { getScoreColor, getScoreLabel } from "../lib/types";

interface Props {
  score: number;
  size?: number;
  label?: string;
}

export default function ScoreGauge({ score, size = 180, label }: Props) {
  const color = getScoreColor(score);
  const radius = 45;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          viewBox="0 0 100 100"
          className="transform -rotate-90"
          style={{ width: size, height: size }}
          role="img"
          aria-label={`Quality score: ${Math.round(score)} out of 100, rated ${getScoreLabel(score)}`}
        >
          <title>Quality score: {Math.round(score)}</title>
          {/* Background ring */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke="var(--color-surface-800)"
            strokeWidth="8"
          />
          {/* Score ring */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{
              animation: "score-fill 1.2s ease-out",
              filter: `drop-shadow(0 0 8px ${color}40)`,
            }}
          />
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="text-4xl font-bold font-[family-name:var(--font-display)]"
            style={{ color }}
          >
            {Math.round(score)}
          </span>
          <span className="text-xs text-surface-300 tracking-wider uppercase mt-0.5">
            {getScoreLabel(score)}
          </span>
        </div>
      </div>

      {label && (
        <span className="text-sm text-surface-300 font-medium">{label}</span>
      )}
    </div>
  );
}

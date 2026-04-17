/* ── ParseIQ API Types ──────────────────────────────── */

export interface APIResponse<T = unknown> {
  success: boolean;
  data: T;
  error: string | null;
}

export interface UploadData {
  job_id: string;
  filename: string;
  file_size: number;
}

export interface JobEvent {
  ts: number;
  level: "info" | "stage" | "error";
  message: string;
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  step: "extract" | "enrich" | "output" | null;
  progress: number;
  error: string | null;
  events: JobEvent[];
  last_event_ts: number;
}

export interface TableSummary {
  name: string;
  record_count: number;
  attribute_count: number;
  quality_score: number;
  anomaly_count: number;
  parent?: string | null;
  children?: string[];
}

export interface AnalysisResult {
  job_id: string;
  tables: string[];
  quality_scores: Record<string, number>;
  anomalies: Record<string, Record<string, string[]>>;
  overall_quality_score: number;
  total_anomalies: number;
  table_summaries: TableSummary[];
  output_files: string[];
  llm_insights: Record<string, unknown> | null;
  raw_metadata: Record<string, unknown>;
  pipeline_info: Record<string, unknown>;
}

export interface AttributeProfile {
  data_type: string;
  total_count: number;
  present_count: number;
  missing_count: number;
  missing_percentage: number;
  unique_values: number;
  quality_score: number;
  anomalies: string[];
  [key: string]: unknown;
}

export interface TableDetail {
  name: string;
  record_count: number;
  quality_score: number;
  attributes: Record<string, AttributeProfile>;
  anomalies: Record<string, string[]>;
  data_preview: Record<string, unknown>[];
  parent?: string | null;
  children?: TableSummary[];
}

/* ── LLM settings (persisted in localStorage) ─────────── */

export type LLMProvider = "openrouter" | "openai" | "azure" | "ollama";

export interface LLMSettings {
  enabled: boolean;
  provider: LLMProvider;
  api_key: string;
  model: string;
  base_url: string;
}

export const DEFAULT_LLM_SETTINGS: LLMSettings = {
  enabled: false,
  provider: "openrouter",
  api_key: "",
  model: "mistralai/mistral-small-3.1-24b-instruct:free",
  base_url: "",
};

const SETTINGS_KEY = "parseiq.llm_settings";

export function loadLlmSettings(): LLMSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_LLM_SETTINGS };
    return { ...DEFAULT_LLM_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return { ...DEFAULT_LLM_SETTINGS };
  }
}

export function saveLlmSettings(s: LLMSettings): void {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
}

/* ── Anomaly Severity ───────────────────────────────── */

export type Severity = "critical" | "high" | "medium" | "low";

export const ANOMALY_SEVERITY: Record<string, Severity> = {
  HIGH_NULL_RATE: "critical",
  MIXED_DATA_TYPES: "critical",
  DUPLICATE_ROWS_DETECTED: "critical",
  NUMERIC_OUTLIERS_DETECTED: "high",
  RANGE_VIOLATION_DETECTED: "high",
  CONSTRAINT_VIOLATION_DETECTED: "high",
  LOW_UNIQUENESS: "medium",
  NEGATIVE_VALUES_DETECTED: "medium",
  PATTERN_INCONSISTENCY: "medium",
  SCALE_VIOLATION_DETECTED: "medium",
  FUTURE_DATE_DETECTED: "low",
  TYPE_CONDITIONAL_FIELD: "low",
};

export function getScoreColor(score: number): string {
  if (score >= 80) return "var(--color-quality-excellent)";
  if (score >= 60) return "var(--color-quality-warning)";
  if (score >= 40) return "var(--color-quality-poor)";
  return "var(--color-quality-critical)";
}

export function getScoreLabel(score: number): string {
  if (score >= 80) return "Excellent";
  if (score >= 60) return "Good";
  if (score >= 40) return "Poor";
  return "Critical";
}

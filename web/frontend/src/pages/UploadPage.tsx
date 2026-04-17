import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { KeyRound, Settings as SettingsIcon, Sparkles, ToggleLeft, ToggleRight, Zap } from "lucide-react";
import FileDropzone from "../components/FileDropzone";
import { uploadFile, startAnalysis } from "../lib/api";
import { loadLlmSettings, saveLlmSettings, type LLMSettings } from "../lib/types";

export default function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [settings, setSettings] = useState<LLMSettings>(loadLlmSettings);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { document.title = "ParseIQ — Upload"; }, []);

  // Refresh settings on mount (user might have come back from /settings)
  useEffect(() => {
    setSettings(loadLlmSettings());
  }, []);

  const toggleLlm = () => {
    const next = { ...settings, enabled: !settings.enabled };
    setSettings(next);
    saveLlmSettings(next);
  };

  const useLlm = settings.enabled;
  const needsKey =
    useLlm && settings.provider !== "ollama" && !settings.api_key.trim();

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const { job_id } = await uploadFile(file, useLlm);
      await startAnalysis(job_id, useLlm ? settings : null);
      navigate(`/processing/${job_id}`);
    } catch (err: unknown) {
      let msg = "Upload failed. Please try again.";
      if (err && typeof err === "object" && "response" in err) {
        const resp = (err as { response?: { data?: { detail?: string; error?: string }; statusText?: string } }).response;
        msg = resp?.data?.detail || resp?.data?.error || resp?.statusText || msg;
      } else if (err instanceof Error) {
        msg = err.message;
      }
      setError(msg);
      setLoading(false);
    }
  };

  return (
    <div className="noise-bg min-h-[calc(100vh-56px)] flex items-center justify-center px-6">
      <div className="w-full max-w-2xl animate-fade-up py-12">
        {/* Hero */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-300 text-xs font-medium mb-6">
            <Zap size={12} />
            AI-Powered Data Quality Analysis
          </div>

          <h1 className="text-4xl md:text-5xl font-bold text-surface-100 tracking-tight font-[family-name:var(--font-display)] leading-tight">
            Profile any dataset
            <br />
            <span className="text-brand-400">in seconds</span>
          </h1>

          <p className="text-surface-300 mt-4 text-lg max-w-md mx-auto">
            Drop a JSON, CSV, XML, or Excel file. Get quality scores,
            anomaly detection, and actionable insights.
          </p>
        </div>

        {/* Dropzone */}
        <div className="animate-fade-up animate-fade-up-delay-1">
          <FileDropzone onFileSelect={setFile} disabled={loading} />
        </div>

        {/* Options */}
        <div className="mt-6 animate-fade-up animate-fade-up-delay-2 space-y-4">
          <div className="flex items-center justify-between rounded-xl border border-surface-300/10 bg-surface-900/60 p-4">
            <button
              onClick={toggleLlm}
              role="switch"
              aria-checked={useLlm}
              aria-label="Toggle LLM enrichment"
              className="flex items-center gap-3 cursor-pointer"
            >
              {useLlm ? (
                <ToggleRight size={26} className="text-brand-400" />
              ) : (
                <ToggleLeft size={26} className="text-surface-300" />
              )}
              <span className="text-left">
                <span className="block text-sm font-semibold text-surface-100">
                  LLM enrichment {useLlm ? "on" : "off"}
                </span>
                <span className="block text-xs text-surface-300 mt-0.5">
                  {useLlm
                    ? `${settings.provider} · ${settings.model.split("/").pop()}`
                    : "Fully local — your data never leaves this machine"}
                </span>
              </span>
            </button>

            <Link
              to="/settings"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-surface-300 hover:text-surface-100 hover:bg-surface-800/60 transition-colors cursor-pointer"
            >
              <SettingsIcon size={12} />
              Configure
            </Link>
          </div>

          {needsKey && (
            <div className="flex items-start gap-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200">
              <KeyRound size={14} className="mt-0.5 flex-shrink-0" />
              <span>
                No API key set for <strong>{settings.provider}</strong>.{" "}
                <Link to="/settings" className="underline hover:text-amber-100">
                  Add your key
                </Link>{" "}
                or turn LLM off to run locally.
              </span>
            </div>
          )}
        </div>

        {/* Action */}
        <div className="mt-6 flex justify-end animate-fade-up animate-fade-up-delay-3">
          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className={
              "flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 cursor-pointer " +
              (file && !loading
                ? "bg-brand-500 text-white hover:bg-brand-600 hover:scale-[1.02] shadow-lg shadow-brand-500/25"
                : "bg-surface-800 text-surface-300/40 cursor-not-allowed")
            }
          >
            <Sparkles size={14} />
            {loading ? "Uploading..." : "Analyze"}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

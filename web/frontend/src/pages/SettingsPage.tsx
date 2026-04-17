import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Check, Eye, EyeOff, KeyRound, Save, Sparkles } from "lucide-react";
import {
  DEFAULT_LLM_SETTINGS,
  loadLlmSettings,
  saveLlmSettings,
  type LLMProvider,
  type LLMSettings,
} from "../lib/types";

const PROVIDER_PRESETS: Record<LLMProvider, { model: string; baseUrl: string; label: string; hint: string }> = {
  openrouter: {
    label: "OpenRouter",
    model: "mistralai/mistral-small-3.1-24b-instruct:free",
    baseUrl: "https://openrouter.ai/api/v1",
    hint: "Any OpenRouter-hosted model. Free-tier keys work.",
  },
  openai: {
    label: "OpenAI",
    model: "gpt-4o-mini",
    baseUrl: "https://api.openai.com/v1",
    hint: "Uses your OpenAI account directly.",
  },
  azure: {
    label: "Azure OpenAI",
    model: "gpt-4o",
    baseUrl: "",
    hint: "Paste your Azure deployment base URL below.",
  },
  ollama: {
    label: "Ollama (local)",
    model: "llama3",
    baseUrl: "http://localhost:11434/v1",
    hint: "Runs fully on your machine. No API key needed.",
  },
};

export default function SettingsPage() {
  const navigate = useNavigate();
  useEffect(() => { document.title = "ParseIQ — Settings"; }, []);
  const [s, setS] = useState<LLMSettings>(loadLlmSettings);
  const [showKey, setShowKey] = useState(false);
  const [saved, setSaved] = useState(false);

  const update = (patch: Partial<LLMSettings>) => setS({ ...s, ...patch });

  const pickProvider = (provider: LLMProvider) => {
    const preset = PROVIDER_PRESETS[provider];
    update({
      provider,
      model: s.provider === provider ? s.model : preset.model,
      base_url: s.provider === provider ? s.base_url : preset.baseUrl,
    });
  };

  const save = () => {
    saveLlmSettings(s);
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  };

  const reset = () => setS({ ...DEFAULT_LLM_SETTINGS });

  const preset = PROVIDER_PRESETS[s.provider];

  return (
    <div className="noise-bg min-h-[calc(100vh-56px)]">
      <div className="max-w-3xl mx-auto px-6 py-12 animate-fade-up">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500/30 to-brand-500/5 border border-brand-500/30 flex items-center justify-center">
            <Sparkles size={18} className="text-brand-300" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-surface-100 tracking-tight font-[family-name:var(--font-display)]">
              LLM settings
            </h1>
            <p className="text-surface-300 text-xs mt-0.5">
              Use your own provider, key, and model. Stored locally in your browser.
            </p>
          </div>
        </div>

        {/* Enable toggle */}
        <div className="mt-8 rounded-xl border border-surface-300/10 bg-surface-900/60 p-5 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-surface-100">Enable LLM enrichment</h3>
            <p className="text-xs text-surface-300 mt-0.5">
              When off, ParseIQ runs 100% locally — your data never leaves this machine.
            </p>
          </div>
          <button
            onClick={() => update({ enabled: !s.enabled })}
            className={`relative w-12 h-6 rounded-full transition-colors cursor-pointer ${
              s.enabled ? "bg-brand-500" : "bg-surface-800"
            }`}
            aria-pressed={s.enabled}
            aria-label="Enable LLM enrichment"
          >
            <span
              className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform ${
                s.enabled ? "translate-x-6" : ""
              }`}
            />
          </button>
        </div>

        {/* Provider picker */}
        <div className="mt-6">
          <label className="block text-xs font-semibold text-surface-200 uppercase tracking-wider mb-2">
            Provider
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {(Object.keys(PROVIDER_PRESETS) as LLMProvider[]).map((p) => {
              const active = s.provider === p;
              return (
                <button
                  key={p}
                  onClick={() => pickProvider(p)}
                  disabled={!s.enabled}
                  className={`relative px-3 py-3 rounded-xl border text-sm font-medium transition-all cursor-pointer text-left ${
                    active
                      ? "border-brand-500/60 bg-brand-500/10 text-brand-200"
                      : "border-surface-300/10 bg-surface-900/60 text-surface-200 hover:border-surface-300/30"
                  } ${!s.enabled ? "opacity-40 cursor-not-allowed" : ""}`}
                >
                  {active && (
                    <span className="absolute top-2 right-2 w-4 h-4 rounded-full bg-brand-500/30 flex items-center justify-center">
                      <Check size={10} className="text-brand-200" />
                    </span>
                  )}
                  {PROVIDER_PRESETS[p].label}
                </button>
              );
            })}
          </div>
          <p className="text-xs text-surface-300 mt-2">{preset.hint}</p>
        </div>

        {/* API key */}
        <div className="mt-6">
          <label className="flex items-center gap-2 text-xs font-semibold text-surface-200 uppercase tracking-wider mb-2">
            <KeyRound size={12} /> API key
          </label>
          <div className="relative">
            <input
              type={showKey ? "text" : "password"}
              value={s.api_key}
              onChange={(e) => update({ api_key: e.target.value })}
              disabled={!s.enabled}
              placeholder={s.provider === "ollama" ? "not required" : "sk-..."}
              className="w-full px-4 py-2.5 pr-11 rounded-lg bg-surface-900/80 border border-surface-300/10 text-sm text-surface-100 font-mono focus:outline-none focus:border-brand-500/50 transition-colors disabled:opacity-40"
            />
            <button
              onClick={() => setShowKey(!showKey)}
              type="button"
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded hover:bg-surface-800/60 text-surface-300 cursor-pointer"
              aria-label={showKey ? "Hide key" : "Show key"}
            >
              {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </div>
        </div>

        {/* Model + base URL */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
          <div>
            <label className="block text-xs font-semibold text-surface-200 uppercase tracking-wider mb-2">
              Model
            </label>
            <input
              type="text"
              value={s.model}
              onChange={(e) => update({ model: e.target.value })}
              disabled={!s.enabled}
              placeholder={preset.model}
              className="w-full px-4 py-2.5 rounded-lg bg-surface-900/80 border border-surface-300/10 text-sm text-surface-100 font-mono focus:outline-none focus:border-brand-500/50 transition-colors disabled:opacity-40"
            />
          </div>
          <div>
            <label className="block text-xs font-semibold text-surface-200 uppercase tracking-wider mb-2">
              Base URL (optional)
            </label>
            <input
              type="text"
              value={s.base_url}
              onChange={(e) => update({ base_url: e.target.value })}
              disabled={!s.enabled}
              placeholder={preset.baseUrl || "https://..."}
              className="w-full px-4 py-2.5 rounded-lg bg-surface-900/80 border border-surface-300/10 text-sm text-surface-100 font-mono focus:outline-none focus:border-brand-500/50 transition-colors disabled:opacity-40"
            />
          </div>
        </div>

        {/* Actions */}
        <div className="mt-8 flex items-center gap-3">
          <button
            onClick={save}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-brand-500 text-white text-sm font-semibold hover:bg-brand-600 transition-colors cursor-pointer shadow-lg shadow-brand-500/20"
          >
            {saved ? <Check size={14} /> : <Save size={14} />}
            {saved ? "Saved" : "Save settings"}
          </button>
          <button
            onClick={reset}
            className="px-5 py-2.5 rounded-xl bg-surface-800 text-surface-200 text-sm hover:bg-surface-800/80 transition-colors cursor-pointer"
          >
            Reset
          </button>
          <button
            onClick={() => navigate("/")}
            className="ml-auto text-sm text-surface-300 hover:text-surface-100 transition-colors cursor-pointer"
          >
            Back to upload →
          </button>
        </div>

        <div className="mt-10 p-4 rounded-xl border border-amber-500/20 bg-amber-500/5 text-xs text-amber-200/80">
          <strong className="text-amber-200">Privacy:</strong> your key never leaves your
          browser except to call your chosen provider through the backend proxy. The
          key is not written to disk.
        </div>
      </div>
    </div>
  );
}

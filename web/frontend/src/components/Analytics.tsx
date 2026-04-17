import { useEffect } from "react";

/**
 * Injects free, privacy-respecting analytics scripts if env vars are set.
 *
 *   VITE_CLARITY_ID   → Microsoft Clarity (free, unlimited, heatmaps + replay)
 *   VITE_GA_ID        → Google Analytics 4 measurement ID (e.g. G-XXXXXXXXXX)
 *
 * Both are optional. If neither is set, nothing is loaded.
 */

const CLARITY_RE = /^[a-zA-Z0-9]+$/;
const GA_RE = /^G-[A-Z0-9]+$/;

export default function Analytics() {
  useEffect(() => {
    const clarityId = import.meta.env.VITE_CLARITY_ID as string | undefined;
    const gaId = import.meta.env.VITE_GA_ID as string | undefined;

    if (clarityId && CLARITY_RE.test(clarityId) && !document.getElementById("ms-clarity")) {
      const s = document.createElement("script");
      s.id = "ms-clarity";
      s.async = true;
      s.src = `https://www.clarity.ms/tag/${encodeURIComponent(clarityId)}`;
      document.head.appendChild(s);
    }

    if (gaId && GA_RE.test(gaId) && !document.getElementById("ga4-loader")) {
      const loader = document.createElement("script");
      loader.id = "ga4-loader";
      loader.async = true;
      loader.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(gaId)}`;
      document.head.appendChild(loader);

      const init = document.createElement("script");
      init.id = "ga4-init";
      init.textContent = [
        "window.dataLayer = window.dataLayer || [];",
        "function gtag(){dataLayer.push(arguments);}",
        "gtag('js', new Date());",
        `gtag('config', ${JSON.stringify(gaId)}, { anonymize_ip: true });`,
      ].join("\n");
      document.head.appendChild(init);
    }
  }, []);

  return null;
}

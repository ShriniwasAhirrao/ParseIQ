/* ── Poll job status until completed/failed ─────────── */

import { useEffect, useRef, useState } from "react";
import { getStatus } from "../lib/api";
import type { JobEvent, JobStatus } from "../lib/types";

export interface PollerState extends JobStatus {
  allEvents: JobEvent[];
}

export function useJobPoller(jobId: string | null, intervalMs = 700) {
  const [state, setState] = useState<PollerState | null>(null);
  const timer = useRef<ReturnType<typeof setInterval>>(undefined);
  const sinceRef = useRef(0);
  const eventsRef = useRef<JobEvent[]>([]);

  useEffect(() => {
    if (!jobId) return;
    sinceRef.current = 0;
    eventsRef.current = [];

    const poll = async () => {
      try {
        const s = await getStatus(jobId, sinceRef.current);
        if (s.events && s.events.length > 0) {
          eventsRef.current = [...eventsRef.current, ...s.events];
          sinceRef.current = s.last_event_ts || sinceRef.current;
        } else if (s.last_event_ts) {
          sinceRef.current = s.last_event_ts;
        }
        setState({ ...s, allEvents: [...eventsRef.current] });
        if (s.status === "completed" || s.status === "failed") {
          clearInterval(timer.current);
        }
      } catch {
        clearInterval(timer.current);
        setState((prev) =>
          prev
            ? { ...prev, status: "failed" as const, error: "Connection lost. Please refresh the page." }
            : null
        );
      }
    };

    poll();
    timer.current = setInterval(poll, intervalMs);
    return () => clearInterval(timer.current);
  }, [jobId, intervalMs]);

  return state;
}

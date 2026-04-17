/* ── ParseIQ API Client ─────────────────────────────── */

import axios from "axios";
import type {
  APIResponse,
  UploadData,
  JobStatus,
  AnalysisResult,
  TableDetail,
  LLMSettings,
} from "./types";

const http = axios.create({ baseURL: "/api", timeout: 120_000 });

export async function uploadFile(
  file: File,
  useLlm: boolean = false
): Promise<UploadData> {
  const form = new FormData();
  form.append("file", file);
  form.append("use_llm", String(useLlm));
  const { data } = await http.post<APIResponse<UploadData>>("/upload", form);
  return data.data;
}

export async function startAnalysis(
  jobId: string,
  settings?: LLMSettings | null
): Promise<void> {
  const body =
    settings && settings.enabled
      ? {
          llm: {
            provider: settings.provider,
            api_key: settings.api_key || undefined,
            model: settings.model || undefined,
            base_url: settings.base_url || undefined,
          },
        }
      : {};
  await http.post(`/analyze/${jobId}`, body);
}

export async function getStatus(
  jobId: string,
  since: number = 0
): Promise<JobStatus> {
  const { data } = await http.get<APIResponse<JobStatus>>(
    `/status/${jobId}`,
    { params: { since } }
  );
  return data.data;
}

export async function getResults(jobId: string): Promise<AnalysisResult> {
  const { data } = await http.get<APIResponse<AnalysisResult>>(
    `/results/${jobId}`
  );
  return data.data;
}

export async function getTableDetail(
  jobId: string,
  tableName: string
): Promise<TableDetail> {
  const { data } = await http.get<APIResponse<TableDetail>>(
    `/results/${jobId}/table/${tableName}`
  );
  return data.data;
}

export function getDownloadUrl(jobId: string): string {
  return `/api/results/${jobId}/download`;
}

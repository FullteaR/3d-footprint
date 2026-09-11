export type DataWarning = { source: string; reason: string };
export type Job = {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  stage: string;
  detail?: string;
  expires_at: number | null;
  warnings: DataWarning[];
};

const fileIds = new WeakMap<Blob, number>();
let nextFileId = 0;
// The exact submitted form identifies a preview. Different File instances with
// identical names/sizes still represent different input; format is independent.
export function formKey(form: FormData): string {
  return JSON.stringify(Array.from(form.entries())
    .filter(([key]) => key !== "fmt")
    .map(([key, value]) => {
      if (typeof value === "string") return [key, value];
      if (!fileIds.has(value)) fileIds.set(value, ++nextFileId);
      return [key, fileIds.get(value)];
    }).sort(([a], [b]) => String(a).localeCompare(String(b))));
}

export async function apiFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const signal = init.signal
    ? AbortSignal.any([init.signal, AbortSignal.timeout(60_000)])
    : AbortSignal.timeout(60_000);
  const response = await fetch(url, { ...init, signal, cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  return response;
}

function pause(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const aborted = () => { clearTimeout(timer); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener("abort", aborted); resolve(); }, 1000);
    signal.addEventListener("abort", aborted, { once: true });
    if (signal.aborted) aborted();
  });
}

export async function waitForJob(id: string, update: (job: Job) => void, signal: AbortSignal): Promise<Job> {
  for (;;) {
    signal.throwIfAborted();
    const job: Job = await (await apiFetch(`/api/jobs/${id}`, { signal })).json();
    signal.throwIfAborted();
    update(job);
    if (job.status === "succeeded") return job;
    if (job.status === "failed" || job.status === "cancelled") {
      throw new Error(job.detail ?? job.status);
    }
    await pause(signal);
  }
}

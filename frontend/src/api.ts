export interface Job {
  id: string;
  model: string;
  goal: string;
  method: string;
  status: 'running' | 'done' | 'failed';
  created: number;
  updated: number;
  run_dir: string;
  error: string;
  log: string;
  result?: string;
}

export interface BackendInfo {
  available: boolean;
  reason: string;
  version: string;
}

export interface Candidate {
  method: string;
  quant: string;
  bits: number;
  score: number;
  fits_target: boolean;
  reasons: string[];
}

export interface PlanPreview {
  model: string;
  params_b: number | null;
  target_vram_gb: number | null;
  license: string | null;
  license_flag: boolean;
  candidates: Candidate[];
}

export interface Attempt {
  n: number;
  method: string;
  quant: string;
  bits: number;
  mixed: boolean;
  ppl: number | null;
  ppl_increase: number | null;
  gate_pass: boolean | null;
  elapsed_s: number;
  note: string;
}

export interface Report {
  baseline: Record<string, unknown> & {
    model: string;
    perplexity: number | null;
    tokens_per_sec: number | null;
    disk_size_gb: number | null;
  };
  quantized: {
    model: string;
    perplexity: number | null;
    tokens_per_sec: number | null;
    disk_size_gb: number | null;
    text_source?: string;
    device?: string;
  };
  quant: { method: string; quant: string; device?: string };
  comparison: { ppl_increase: number | null; gate_pass: boolean | null };
  attempts: Attempt[];
}

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json() as Promise<T>;
}

export const api = {
  jobs: () => fetch('/jobs').then(j<Job[]>),
  job: (id: string) => fetch(`/jobs/${id}`).then(j<Job>),
  report: (id: string) => fetch(`/jobs/${id}/report`).then(j<Report>),
  backends: () => fetch('/api/backends').then(j<Record<string, BackendInfo>>),
  plan: (model: string, goal: string) =>
    fetch(`/api/plan?model=${encodeURIComponent(model)}&goal=${goal}`).then(j<PlanPreview>),
  submit: (p: { model: string; goal: string; method: string; max_samples: number; max_attempts: number; bits: number }) =>
    fetch('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(p),
    }).then(j<{ id: string; status: string }>),
};

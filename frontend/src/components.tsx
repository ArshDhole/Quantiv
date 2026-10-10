import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { BackendInfo, Candidate, Job, PlanPreview, Report } from './api';
import { GOALS, METHODS, MODELS } from './models';

/* ---------- atoms ---------- */

export function Badge({ status }: { status: string }) {
  const cls = status === 'done' ? 'b-done' : status === 'failed' ? 'b-fail' : 'b-run';
  return <span className={`badge ${cls}`}>{status}</span>;
}

export function Topbar({ backends }: { backends: Record<string, BackendInfo> }) {
  const keys = Object.keys(backends);
  const n = keys.filter((k) => backends[k].available).length;
  return (
    <div className="topbar">
      <div className="mark">Q</div>
      <div className="brand">
        Quantiv<small>PRECISION LAB</small>
      </div>
      <div className="pills">
        <span className="pill">
          <span className="dotlive">●</span>&nbsp;<b>{keys.length ? `${n}/${keys.length}` : '…'}</b>&nbsp;backends
        </span>
        <a className="pill" href="https://github.com/ArshDhole/Quantiv">
          GitHub →
        </a>
      </div>
    </div>
  );
}

export function Hero() {
  return (
    <div className="hero">
      <div className="eyebrow">EVERY BIT, ACCOUNTED FOR</div>
      <h1>
        Smaller models. <em>Receipts included.</em>
      </h1>
      <p className="lede">
        Pick a model and a goal — Quantiv quantizes it, measures both versions, and serves the
        verdict.
      </p>
      <div className="chips">
        <span className="chip"><b>05</b> backends</span>
        <span className="chip"><b>44</b> tests green</span>
        <span className="chip"><b>$0</b> runs offline</span>
      </div>
    </div>
  );
}

/* ---------- composer ---------- */

interface ComposerProps {
  model: string;
  setModel: (m: string) => void;
  custom: boolean;
  setCustom: (c: boolean) => void;
  goal: string;
  setGoal: (g: string) => void;
  method: string;
  setMethod: (m: string) => void;
  samples: string;
  setSamples: (s: string) => void;
  attempts: string;
  setAttempts: (s: string) => void;
  plan: PlanPreview | null;
  busy: boolean;
  onFire: () => void;
}

const GROUPS: Array<[string, number, number]> = [
  ['Small & fast', 0, 3],
  ['Qwen chat', 3, 7],
  ['Popular', 7, 14],
];

export function Composer(p: ComposerProps) {
  const fam = MODELS.find((m) => m.id === p.model);
  return (
    <div className="card">
      <h2>New run</h2>
      <div className="frow main">
        <div>
          <label>Model</label>
          <select
            value={p.custom ? '__custom' : p.model}
            onChange={(e) => {
              const v = e.target.value;
              p.setCustom(v === '__custom');
              if (v !== '__custom') p.setModel(v);
            }}
          >
            {GROUPS.map((g) => (
              <optgroup label={g[0]} key={g[0]}>
                {MODELS.slice(g[1], g[2]).map((m) => (
                  <option value={m.id} key={m.id}>
                    {m.label}
                  </option>
                ))}
              </optgroup>
            ))}
            <option value="__custom">Custom HF id or local path…</option>
          </select>
          {p.custom && (
            <div style={{ marginTop: 8 }}>
              <input
                value={p.model}
                placeholder="org/model-name or /local/path"
                onChange={(e) => p.setModel(e.target.value)}
              />
            </div>
          )}
          <div className="hint" style={{ marginTop: 6 }}>
            {fam ? `family ${fam.family} · ${fam.label}` : 'Any public HF repo id or a local model directory.'}
          </div>
        </div>
        <div>
          <label>Engine</label>
          <select value={p.method} onChange={(e) => p.setMethod(e.target.value)}>
            <option value="auto">Auto — planner picks</option>
            {METHODS.filter((m) => m !== 'auto').map((m) => (
              <option value={m} key={m}>{m}</option>
            ))}
          </select>
        </div>
        <div>
          <button onClick={p.onFire} disabled={p.busy || !p.model}>
            {p.busy ? 'Running…' : 'Quantize →'}
          </button>
        </div>
      </div>
      <div style={{ marginTop: 14 }}>
        <label>Goal</label>
        <div className="goals">
          {GOALS.map((g) => (
            <div
              key={g[0]}
              className={'goal' + (p.goal === g[0] ? ' sel' : '')}
              onClick={() => p.setGoal(g[0])}
            >
              <b>{g[1]}</b>
              <small>{g[2]}</small>
            </div>
          ))}
        </div>
        <PlanStrip plan={p.plan} />
      </div>
      <div className="frow sub">
        <div>
          <label>Eval samples</label>
          <select value={p.samples} onChange={(e) => p.setSamples(e.target.value)}>
            <option>8</option>
            <option>16</option>
            <option>32</option>
          </select>
        </div>
        <div>
          <label>Max attempts</label>
          <select value={p.attempts} onChange={(e) => p.setAttempts(e.target.value)}>
            <option>1</option>
            <option>2</option>
            <option>3</option>
            <option>4</option>
            <option>5</option>
          </select>
        </div>
      </div>
    </div>
  );
}

function PlanStrip({ plan }: { plan: PlanPreview | null }) {
  if (!plan) return <div className="planline">price check loading…</div>;
  const top = plan.candidates[0];
  return (
    <div className="planline">
      top pick: <b>{top ? `${top.method} ${top.quant}` : '?'}</b>
      {plan.params_b ? ` · ${plan.params_b.toFixed(2)}B params` : ''}
      {plan.license ? ` · ${plan.license}` : ''}
      <table>
        <tbody>
          <tr>
            <th>Engine</th>
            <th>Quant</th>
            <th>Score</th>
            <th>Fit</th>
          </tr>
          {plan.candidates.slice(0, 3).map((c: Candidate) => (
            <tr key={c.method + c.quant}>
              <td>{c.method}</td>
              <td>{c.quant}</td>
              <td>{c.score}</td>
              <td>{c.fits_target ? 'fits' : 'tight'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- live run ---------- */

const STEPS = ['Analyze', 'Quantize', 'Evaluate', 'Report'];

export function stepFromLog(log: string): number {
  const L = (log || '').toLowerCase();
  if (/report\.json/.test(L)) return 4;
  if (/gate=|attempt \d/.test(L)) return 3;
  if (/quantiz|calibrat/.test(L)) return 2;
  if (/baseline|analyz|warming|accepted/.test(L)) return 1;
  return 0;
}

export function fmtElapsed(created: number): string {
  const ago = Math.max(0, Date.now() / 1000 - created);
  const mm = Math.floor(ago / 60);
  const ss = Math.floor(ago % 60).toString().padStart(2, '0');
  return `elapsed ${mm}:${ss} · small models typically take 5–15 min`;
}

function hintFor(log: string): string {
  if (/attempt \d/.test(log)) return 'quantizing + evaluating this attempt…';
  if (/baseline/.test(log)) return 'baseline measured · starting attempts…';
  return 'warming up… first progress lands after the baseline loads (1–3 min on small models).';
}

export function ActiveRun({ job }: { job: Job }) {
  const done = stepFromLog(job.log || '');
  return (
    <div className="card">
      <h2>Live run</h2>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <b>Running · {job.model}</b>
        <Badge status={job.status} />
        <span className="hint" style={{ marginLeft: 'auto' }}>
          {fmtElapsed(job.created)}
        </span>
      </div>
      <div className="pbar"><span /></div>
      <div className="stepsline">
        {STEPS.map((s, i) => (
          <span key={s} className={i < done ? 'done' : i === done ? 'on' : ''} title={s} />
        ))}
      </div>
      <div className="hint">{hintFor(job.log || '')}</div>
      <div className="mono" style={{ maxHeight: 180, marginTop: 8 }}>
        {job.log || '…'}
      </div>
    </div>
  );
}

/* ---------- ledger + certificate ---------- */

export function Ledger({
  jobs,
  onOpen,
}: {
  jobs: Job[];
  onOpen: (id: string) => void;
}) {
  const short = (m: string) => {
    const p = m.split('/');
    return p.length > 1 ? p[1] : m;
  };
  return (
    <div className="card">
      <h2>Runs</h2>
      <table>
        <tbody>
          <tr>
            <th>Run</th>
            <th>Model</th>
            <th>Goal</th>
            <th>Status</th>
            <th>Result</th>
            <th></th>
          </tr>
          {jobs.length === 0 && (
            <tr>
              <td colSpan={6} className="hint">
                no runs yet — fire one above
              </td>
            </tr>
          )}
          {jobs.map((j) => (
            <tr key={j.id}>
              <td className="hint">{j.id}</td>
              <td>{short(j.model)}</td>
              <td>{j.goal}</td>
              <td><Badge status={j.status} /></td>
              <td className="hint">{j.result || ''}</td>
              <td><a onClick={() => onOpen(j.id)}>open →</a></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Certificate({ job, report }: { job: Job; report: Report | null }) {
  if (job.status === 'failed') {
    return (
      <div className="card">
        <h2>Result</h2>
        <h3>
          {job.model} <Badge status={job.status} />
        </h3>
        <p className="fail">{job.error || 'failed'}</p>
        {job.log && (
          <>
            <h3>Log</h3>
            <div className="mono">{job.log}</div>
          </>
        )}
      </div>
    );
  }
  if (!report) {
    return (
      <div className="card">
        <h2>Result</h2>
        <h3>
          {job.model} <Badge status={job.status} />
        </h3>
        {job.log && (
          <>
            <h3>Log</h3>
            <div className="mono">{job.log}</div>
          </>
        )}
      </div>
    );
  }
  const q = report.quantized;
  const c = report.comparison;
  const base = report.baseline;
  const pass = c.gate_pass === true;
  const baseDisk = base.disk_size_gb as number | null;
  const shrink =
    baseDisk && q.disk_size_gb && q.disk_size_gb > 0
      ? `${(baseDisk / q.disk_size_gb).toFixed(1)}× smaller`
      : '—';
  const pplDelta =
    typeof c.ppl_increase === 'number' ? `${c.ppl_increase >= 0 ? '+' : ''}${(c.ppl_increase * 100).toFixed(1)}%` : '—';
  const row = (label: string, b: string, a: string) => (
    <tr key={label}>
      <td>{label}</td>
      <td>{b}</td>
      <td>{a}</td>
    </tr>
  );
  return (
    <div className="card">
      <h2>Result</h2>
      <h3>
        {job.model}{' '}
        <small className="hint">
          {job.id} · {job.goal} · {job.method}
        </small>{' '}
        <Badge status={job.status} />
      </h3>
      <div className="certgrid" style={{ margin: '14px 0' }}>
        <div><small>GATE</small><b className={pass ? 'pass' : 'fail'}>{pass ? 'PASS' : 'FAIL'}</b></div>
        <div><small>METHOD</small><b>{report.quant.method} {report.quant.quant}</b></div>
        <div><small>SIZE</small><b>{shrink}</b></div>
        <div><small>SPEED</small><b>{q.tokens_per_sec ?? '—'} tok/s</b></div>
        <div><small>TEXTS</small><b>{(q.text_source || '').split('(')[0]}</b></div>
        <div><small>DEVICE</small><b>{q.device || report.quant.device || ''}</b></div>
      </div>
      <p className="hint">
        measured, not claimed ·{' '}
        <a href={`/jobs/${job.id}/report`} target="_blank" rel="noreferrer">
          full report.json →
        </a>
      </p>
      <h3>Before → after</h3>
      <table>
        <tbody>
          <tr><th>Metric</th><th>Baseline</th><th>Quantized</th></tr>
          {row('Perplexity (↓ better)', String(base.perplexity ?? '—'), String(q.perplexity ?? '—') + ` (${pplDelta})`)}
          {row('Disk size', baseDisk != null ? `${baseDisk} GB` : '—', q.disk_size_gb != null ? `${q.disk_size_gb} GB` : '—')}
          {row('Decode speed', (base.tokens_per_sec as number | null) != null ? `${base.tokens_per_sec} tok/s` : '—', q.tokens_per_sec != null ? `${q.tokens_per_sec} tok/s` : '—')}
        </tbody>
      </table>
      <p>
        <a href={`/jobs/${job.id}/download`}>
          <button style={{ width: 'auto', marginTop: 6 }}>Download quantized model (.zip) ↓</button>
        </a>
      </p>
      {report.attempts.length > 0 && (
        <table>
          <tbody>
            <tr><th>#</th><th>Method</th><th>Quant</th><th>PPL</th><th>Gate</th></tr>
            {report.attempts.map((a) => (
              <tr key={a.n}>
                <td>{a.n}</td><td>{a.method}</td><td>{a.quant}</td><td>{a.ppl}</td>
                <td>{a.gate_pass ? 'PASS' : 'FAIL'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {job.log && (
        <>
          <h3>Log</h3>
          <div className="mono">{job.log}</div>
        </>
      )}
    </div>
  );
}

/* ---------- tradeoff chart ---------- */

export interface TradePoint {
  name: string;
  ppl: number;
  disk: number;
}

export function TradeoffChart({ points }: { points: TradePoint[] }) {
  if (points.length === 0) {
    return (
      <div className="card">
        <h2>Quality vs size</h2>
        <p className="hint">
          Completed runs will appear here as perplexity-change vs disk size — the tradeoff,
          plotted from measured reports.
        </p>
      </div>
    );
  }
  return (
    <div className="card">
      <h2>Quality vs size — measured</h2>
      <div className="chartbox" style={{ height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid stroke="#e4ded2" />
            <XAxis
              type="number"
              dataKey="ppl"
              name="PPL increase"
              tick={{ fontSize: 11 }}
              label={{ value: 'PPL increase', position: 'insideBottom', offset: -12, fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="disk"
              name="Disk GB"
              tick={{ fontSize: 11 }}
              label={{ value: 'Disk GB', angle: -90, position: 'insideLeft', fontSize: 11 }}
            />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name="runs" data={points} fill="#0e7c6b" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="hint">lower-left is better · each point is a finished run's measured report</p>
    </div>
  );
}

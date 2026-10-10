import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type BackendInfo, type Job, type PlanPreview, type Report } from './api';
import {
  ActiveRun,
  Certificate,
  Composer,
  Hero,
  Ledger,
  Topbar,
  TradeoffChart,
  type TradePoint,
} from './components';
import { shortModel } from './models';

export default function App() {
  const [model, setModel] = useState('HuggingFaceTB/SmolLM2-135M');
  const [custom, setCustom] = useState(false);
  const [goal, setGoal] = useState('balanced');
  const [method, setMethod] = useState('auto');
  const [samples, setSamples] = useState('16');
  const [attempts, setAttempts] = useState('3');
  const [bits, setBits] = useState('4');
  const [plan, setPlan] = useState<PlanPreview | null>(null);
  const [backends, setBackends] = useState<Record<string, BackendInfo>>({});
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Job | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [busy, setBusy] = useState(false);
  const [points, setPoints] = useState<TradePoint[]>([]);
  const timer = useRef<number>(0);

  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await api.jobs());
    } catch {
      /* offline */
    }
  }, []);

  // price preview (debounced)
  useEffect(() => {
    if (!model) {
      setPlan(null);
      return;
    }
    const t = window.setTimeout(async () => {
      try {
        setPlan(await api.plan(model, goal));
      } catch {
        setPlan(null);
      }
    }, 400);
    return () => window.clearTimeout(t);
  }, [model, goal]);

  useEffect(() => {
    api.backends().then(setBackends).catch(() => undefined);
    refreshJobs();
    const t = window.setInterval(() => refreshJobs(), 4000);
    return () => window.clearInterval(t);
  }, [refreshJobs]);

  // detail polling for the selected job
  useEffect(() => {
    if (!selected) {
      setDetail(null);
      setReport(null);
      return;
    }
    let stop = false;
    const poll = async () => {
      try {
        const j = await api.job(selected);
        if (stop) return;
        setDetail(j);
        if (j.status === 'done') {
          try {
            setReport(await api.report(selected));
          } catch {
            /* not ready */
          }
          window.clearInterval(timer.current);
        }
      } catch {
        /* gone */
      }
    };
    poll();
    timer.current = window.setInterval(poll, 2000);
    return () => {
      stop = true;
      window.clearInterval(timer.current);
    };
  }, [selected]);

  // chart points from finished runs
  useEffect(() => {
    const done = jobs.filter((j) => j.status === 'done');
    if (done.length === 0) {
      setPoints([]);
      return;
    }
    let stop = false;
    (async () => {
      const pts: TradePoint[] = [];
      for (const j of done.slice(0, 20)) {
        try {
          const r = await api.report(j.id);
          if (typeof r.comparison.ppl_increase === 'number' && r.quantized.disk_size_gb) {
            pts.push({
              name: `${shortModel(j.model)} ${r.quant.method}`,
              ppl: Math.round(r.comparison.ppl_increase * 1000) / 10,
              disk: r.quantized.disk_size_gb,
            });
          }
        } catch {
          /* skip */
        }
        if (stop) return;
      }
      if (!stop) setPoints(pts);
    })();
    return () => {
      stop = true;
    };
  }, [jobs]);

  // elapsed ticker for the live card
  const [, setTick] = useState(0);
  useEffect(() => {
    const t = window.setInterval(() => setTick((x) => x + 1), 1000);
    return () => window.clearInterval(t);
  }, []);

  const fire = async () => {
    setBusy(true);
    try {
      const r = await api.submit({
        model,
        goal,
        method,
        max_samples: parseInt(samples || '16', 10),
        max_attempts: parseInt(attempts || '3', 10),
        bits: parseInt(bits || '4', 10),
      });
      await refreshJobs();
      setSelected(r.id);
      document.getElementById('ledger-anchor')?.scrollIntoView({ behavior: 'smooth' });
    } finally {
      setBusy(false);
    }
  };

  const active = jobs.find((j) => j.status === 'running') ?? null;

  return (
    <div className="wrap">
      <Topbar backends={backends} />

      <Hero />

      <Composer
        model={model}
        setModel={setModel}
        custom={custom}
        setCustom={setCustom}
        goal={goal}
        setGoal={setGoal}
        method={method}
        setMethod={setMethod}
        samples={samples}
        setSamples={setSamples}
        attempts={attempts}
        setAttempts={setAttempts}
        bits={bits}
        setBits={setBits}
        plan={plan}
        busy={busy}
        onFire={fire}
      />

      {active && <ActiveRun job={active} />}

      <TradeoffChart points={points} />

      <div id="ledger-anchor">
        <Ledger jobs={jobs} onOpen={setSelected} />
      </div>

      {detail && (
        <Certificate
          job={detail}
          report={detail.status === 'done' ? report : null}
        />
      )}

      <footer>Quantiv · every number measured</footer>
    </div>
  );
}

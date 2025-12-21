"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  BadgeCheck,
  BookOpen,
  Brain,
  Database,
  Download,
  FileUp,
  Play,
  ShieldAlert,
} from "lucide-react";

type TxnType = "CASH_IN" | "CASH_OUT" | "DEBIT" | "PAYMENT" | "TRANSFER";

type PredictResponse = {
  risk_score: number;
  risk_bucket: "low" | "medium" | "high";
  recommended_action: "approve" | "review" | "block";
  model_version?: string | null;
  thresholds_version?: string | null;
};

type BatchPredictResponse = {
  n_rows: number;
  n_success: number;
  n_failed: number;
  results: Array<Record<string, any>>;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

function clamp01(x: number) {
  if (Number.isNaN(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function fmtPct(x: number) {
  return `${Math.round(clamp01(x) * 100)}%`;
}

type TabKey = "description" | "execute";

function TabButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "rounded-full px-4 py-2 text-sm font-medium transition",
        "backdrop-blur border",
        active
          ? "bg-black text-white border-black shadow-sm"
          : "bg-white/70 text-gray-800 border-gray-200 hover:bg-white",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

function BentoCard({
  title,
  icon,
  children,
  className = "",
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={[
        "rounded-3xl border border-gray-200 bg-white/75 shadow-sm backdrop-blur",
        className,
      ].join(" ")}
    >
      <div className="flex items-center gap-2 border-b border-gray-100 px-6 py-4">
        {icon ? <span className="text-gray-700">{icon}</span> : null}
        <h2 className="text-sm font-semibold tracking-tight text-gray-900">{title}</h2>
      </div>
      <div className="px-6 py-5">{children}</div>
    </section>
  );
}

function Pill({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <span className="inline-flex items-center rounded-full border border-gray-200 bg-white/70 px-3 py-1 text-xs text-gray-800">
      {children}
    </span>
  );
}

export default function Home() {
  // ---- Tabs ----
  const [tab, setTab] = useState<TabKey>("execute");

  // ---- Raw form state ----
  const [step, setStep] = useState<number>(180);
  const [type, setType] = useState<TxnType>("CASH_OUT");
  const [amount, setAmount] = useState<number>(114664.45);

  const [oldbalanceOrg, setOldbalanceOrg] = useState<number>(0);
  const [newbalanceOrig, setNewbalanceOrig] = useState<number>(0);
  const [oldbalanceDest, setOldbalanceDest] = useState<number>(163667.97);
  const [newbalanceDest, setNewbalanceDest] = useState<number>(278332.42);

  const [nameOrig, setNameOrig] = useState<string>("C1911904419");
  const [nameDest, setNameDest] = useState<string>("C763019604");

  // ---- UI state ----
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // ---- Batch state ----
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchPredictResponse | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);

  const isInvalid = useMemo(() => {
    if (step < 0) return true;
    if (amount < 0) return true;
    if (oldbalanceOrg < 0 || newbalanceOrig < 0 || oldbalanceDest < 0 || newbalanceDest < 0)
      return true;
    return false;
  }, [step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]);

  async function handleScore() {
    setError(null);
    setResult(null);

    if (isInvalid) {
      setError("Invalid input: numbers cannot be negative.");
      return;
    }

    setLoading(true);
    const t0 = performance.now();

    try {
      const payload = {
        raw: {
          step,
          type,
          amount,
          oldbalanceOrg,
          newbalanceOrig,
          oldbalanceDest,
          newbalanceDest,
          nameOrig: nameOrig?.trim() || null,
          nameDest: nameDest?.trim() || null,
        },
      };

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const text = await res.text();
      let data: any;

      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(`API returned non-JSON: ${text}`);
      }

      if (!res.ok) {
        throw new Error(data?.detail ? String(data.detail) : `HTTP ${res.status}`);
      }

      setResult(data as PredictResponse);
    } catch (e: any) {
      setError(e?.message || "Unknown error");
    } finally {
      setLoading(false);
      const t1 = performance.now();
      console.log("[PWA] /predict latency_ms:", Math.round(t1 - t0));
    }
  }

  async function handleBatchUpload() {
    setBatchError(null);
    setBatchResult(null);

    if (!batchFile) {
      setBatchError("Please select a CSV file first.");
      return;
    }

    if (!batchFile.name.toLowerCase().endsWith(".csv")) {
      setBatchError("File must be a .csv");
      return;
    }

    setBatchLoading(true);

    try {
      const form = new FormData();
      form.append("file", batchFile);

      const res = await fetch(`${API_BASE}/predict_batch?mode=raw`, {
        method: "POST",
        body: form,
      });

      const text = await res.text();
      let data: any;

      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(`API returned non-JSON: ${text}`);
      }

      if (!res.ok) {
        throw new Error(data?.detail ? String(data.detail) : `HTTP ${res.status}`);
      }

      setBatchResult(data as BatchPredictResponse);
    } catch (e: any) {
      setBatchError(e?.message || "Batch error");
    } finally {
      setBatchLoading(false);
    }
  }

  const scoreBar = result ? `${Math.round(clamp01(result.risk_score) * 100)}%` : "0%";
  const scorePct = result ? fmtPct(result.risk_score) : "0%";

  return (
    <main className="relative min-h-screen text-gray-900">
      {/* Background base */}
      <div className="fixed inset-0 -z-20 bg-gray-50" />

      {/* Mesh gradient SVG */}
      <div
        className="fixed inset-0 -z-20 opacity-90"
        style={{
          backgroundImage: "url(/assets/mesh.svg)",
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      />

      {/* Noise overlay via CSS utility */}
      <div className="fixed inset-0 -z-10 bg-noise opacity-[0.10] mix-blend-multiply pointer-events-none" />

      <div className="mx-auto max-w-6xl px-4 py-10">
        {/* Hero */}
        <header className="rounded-3xl border border-white/60 bg-white/55 px-6 py-7 shadow-sm backdrop-blur">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-xs font-semibold tracking-wide text-gray-700">
                ML SYSTEM DEMO • REAL-TIME RISK SCORING
              </p>
              <h1 className="mt-2 text-3xl font-bold tracking-tight">PaySim Fraud Risk</h1>
              <p className="mt-2 text-sm text-gray-700">
                Real-time scoring and upload sample data.
              </p>
              <p className="mt-1 text-xs text-gray-600">
                API Base: <span className="font-mono">{API_BASE}</span>
              </p>

              <div className="mt-4 flex flex-wrap gap-2">
                <Pill>HGB v2</Pill>
                <Pill>FastAPI /predict</Pill>
                <Pill>Risk, Bucket, and Action</Pill>
              </div>
            </div>

            <div className="flex gap-2">
              <TabButton active={tab === "description"} onClick={() => setTab("description")}>
                Description
              </TabButton>
              <TabButton active={tab === "execute"} onClick={() => setTab("execute")}>
                Execute
              </TabButton>
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="mt-8">
          {tab === "description" && (
            <div className="grid gap-6 md:grid-cols-12">
              <BentoCard
                title="Project & Problem"
                icon={<BookOpen className="h-4 w-4" />}
                className="md:col-span-7"
              >
                <p className="text-sm leading-6 text-gray-700">
                  Fraudulent transactions are rare but expensive. This project provides a real-time
                  risk scoring service that predicts fraud probability per transaction and maps it
                  into an operational decision: <b>approve</b>, <b>review</b>, or <b>block</b>.
                </p>

                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  <div className="rounded-2xl border border-gray-200 bg-white/70 p-4">
                    <div className="flex items-center gap-2 text-sm font-semibold text-gray-900">
                      <Activity className="h-4 w-4" />
                      Real-time API
                    </div>
                    <p className="mt-2 text-sm text-gray-700">
                      Single transaction scoring via <span className="font-mono">/predict</span>.
                    </p>
                  </div>

                  <div className="rounded-2xl border border-gray-200 bg-white/70 p-4">
                    <div className="flex items-center gap-2 text-sm font-semibold text-gray-900">
                      <BadgeCheck className="h-4 w-4" />
                      Decision-ready output
                    </div>
                    <p className="mt-2 text-sm text-gray-700">
                      Scoring into bucket (low/medium/high).
                    </p>
                  </div>
                </div>
              </BentoCard>

              <BentoCard
                title="Modeling Approach"
                icon={<Brain className="h-4 w-4" />}
                className="md:col-span-5"
              >
                <ul className="list-disc space-y-2 pl-5 text-sm text-gray-700">
                  <li>
                    <b>Model v1:</b> baseline model for stable pipeline reference.
                  </li>
                  <li>
                    <b>Model v2 (final):</b> <b>HistGradientBoosting (HGB)</b> to capture non-linear
                    patterns and interactions.
                  </li>
                </ul>
                <p className="mt-3 text-sm text-gray-600">
                  Outputs are converted into risk buckets and recommended actions.
                </p>
              </BentoCard>

              <BentoCard
                title="Dataset Columns (PaySim subset)"
                icon={<Database className="h-4 w-4" />}
                className="md:col-span-12"
              >
                <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-gray-50">
                      <tr className="border-b border-gray-200">
                        <th className="p-3">Column</th>
                        <th className="p-3">Meaning</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        ["step", "Time step index (hour) in the simulation"],
                        ["type", "Transaction type (CASH_IN, CASH_OUT, etc.)"],
                        ["amount", "Transaction amount"],
                        ["oldbalanceOrg", "Sender balance before the transaction"],
                        ["newbalanceOrig", "Sender balance after the transaction"],
                        ["oldbalanceDest", "Receiver balance before the transaction"],
                        ["newbalanceDest", "Receiver balance after the transaction"],
                      ].map(([k, v]) => (
                        <tr key={k} className="border-b border-gray-100">
                          <td className="p-3 font-mono text-xs">{k}</td>
                          <td className="p-3 text-gray-700">{v}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </BentoCard>
            </div>
          )}

          {tab === "execute" && (
            <div className="grid gap-6 md:grid-cols-12">
              {/* Form */}
              <BentoCard
                title="Score a Transaction"
                icon={<Play className="h-4 w-4" />}
                className="md:col-span-7"
              >
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <div>
                    <label className="text-sm font-medium">Time step (hour index)</label>
                    <input
                      type="number"
                      className="mt-1 w-full rounded-2xl border border-gray-200 bg-white px-3 py-2.5"
                      value={step}
                      onChange={(e) => setStep(Number(e.target.value))}
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium">Transaction type</label>
                    <select
                      className="mt-1 w-full rounded-2xl border border-gray-200 bg-white px-3 py-2.5"
                      value={type}
                      onChange={(e) => setType(e.target.value as TxnType)}
                    >
                      <option value="CASH_IN">CASH_IN</option>
                      <option value="CASH_OUT">CASH_OUT</option>
                      <option value="DEBIT">DEBIT</option>
                      <option value="PAYMENT">PAYMENT</option>
                      <option value="TRANSFER">TRANSFER</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Transaction amount</label>
                    <input
                      type="number"
                      className="mt-1 w-full rounded-2xl border border-gray-200 bg-white px-3 py-2.5"
                      value={amount}
                      onChange={(e) => setAmount(Number(e.target.value))}
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium">Receiver ID (optional)</label>
                    <input
                      className="mt-1 w-full rounded-2xl border border-gray-200 bg-white px-3 py-2.5 font-mono text-sm"
                      value={nameDest}
                      onChange={(e) => setNameDest(e.target.value)}
                      placeholder="Cxxxx or Mxxxx"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium">Sender balance (before)</label>
                    <input
                      type="number"
                      className="mt-1 w-full rounded-2xl border border-gray-200 bg-white px-3 py-2.5"
                      value={oldbalanceOrg}
                      onChange={(e) => setOldbalanceOrg(Number(e.target.value))}
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium">Sender balance (after)</label>
                    <input
                      type="number"
                      className="mt-1 w-full rounded-2xl border border-gray-200 bg-white px-3 py-2.5"
                      value={newbalanceOrig}
                      onChange={(e) => setNewbalanceOrig(Number(e.target.value))}
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium">Receiver balance (before)</label>
                    <input
                      type="number"
                      className="mt-1 w-full rounded-2xl border border-gray-200 bg-white px-3 py-2.5"
                      value={oldbalanceDest}
                      onChange={(e) => setOldbalanceDest(Number(e.target.value))}
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium">Receiver balance (after)</label>
                    <input
                      type="number"
                      className="mt-1 w-full rounded-2xl border border-gray-200 bg-white px-3 py-2.5"
                      value={newbalanceDest}
                      onChange={(e) => setNewbalanceDest(Number(e.target.value))}
                    />
                  </div>

                  <div className="sm:col-span-2">
                    <label className="text-sm font-medium">Sender ID (optional)</label>
                    <input
                      className="mt-1 w-full rounded-2xl border border-gray-200 bg-white px-3 py-2.5 font-mono text-sm"
                      value={nameOrig}
                      onChange={(e) => setNameOrig(e.target.value)}
                      placeholder="Cxxxx"
                    />
                  </div>
                </div>

                <div className="mt-5 flex flex-wrap items-center gap-3">
                  <button
                    onClick={handleScore}
                    disabled={loading || isInvalid}
                    className="inline-flex items-center gap-2 rounded-2xl bg-black px-4 py-2.5 text-sm font-medium text-white disabled:opacity-50"
                  >
                    <Play className="h-4 w-4" />
                    {loading ? "Scoring..." : "Score Transaction"}
                  </button>

                  {isInvalid && (
                    <span className="inline-flex items-center gap-2 text-sm text-red-600">
                      <ShieldAlert className="h-4 w-4" />
                      Invalid input: numbers cannot be negative.
                    </span>
                  )}
                </div>

                {error && (
                  <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                    {error}
                  </div>
                )}
              </BentoCard>

              {/* Result */}
              <BentoCard
                title="Result"
                icon={<Activity className="h-4 w-4" />}
                className="md:col-span-5"
              >
                {!result ? (
                  <p className="text-sm text-gray-700">
                    Run a score to see the risk output here.
                  </p>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <div className="text-sm text-gray-600">Risk Score</div>
                      <div className="mt-1 text-3xl font-bold tracking-tight">
                        {result.risk_score.toFixed(6)}
                      </div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        <Pill>
                          bucket: <b className="ml-1">{result.risk_bucket}</b>
                        </Pill>
                        <Pill>
                          action: <b className="ml-1">{result.recommended_action}</b>
                        </Pill>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-gray-200 bg-white/70 p-4">
                      <div className="flex items-center justify-between text-xs text-gray-600">
                        <span>risk ≈ {scorePct}</span>
                        <span className="font-mono">
                          model {result.model_version ?? "-"} • thr {result.thresholds_version ?? "-"}
                        </span>
                      </div>
                      <div className="mt-3 h-2.5 w-full rounded-full bg-gray-100">
                        <div
                          className="h-2.5 rounded-full bg-black"
                          style={{ width: scoreBar }}
                          title={scorePct}
                        />
                      </div>
                    </div>
                  </div>
                )}
              </BentoCard>

              {/* Batch */}
              <BentoCard
                title="Batch (CSV Upload)"
                icon={<FileUp className="h-4 w-4" />}
                className="md:col-span-12"
              >
                <p className="text-sm text-gray-700">
                  Please upload the provided sample data to trying the models.
                </p>

                <div className="mt-4 flex flex-wrap items-center gap-3">
                  <a
                    href="/demo_batch_raw.csv"
                    download
                    className="inline-flex items-center gap-2 rounded-2xl border border-gray-200 bg-white/80 px-3 py-2 text-sm font-medium text-gray-900 hover:bg-white"
                  >
                    <Download className="h-4 w-4" />
                    Download Demo CSV
                  </a>

                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setBatchFile(e.target.files?.[0] ?? null)}
                    className="block text-sm"
                  />

                  <button
                    onClick={handleBatchUpload}
                    disabled={batchLoading}
                    className="inline-flex items-center gap-2 rounded-2xl bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
                  >
                    <FileUp className="h-4 w-4" />
                    {batchLoading ? "Uploading..." : "Upload & Score"}
                  </button>
                </div>

                {batchError && (
                  <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                    {batchError}
                  </div>
                )}

                {batchResult && (
                  <div className="mt-5 rounded-2xl border border-gray-200 bg-white/75 p-5">
                    <div className="text-sm text-gray-700">
                      rows: <b>{batchResult.n_rows}</b> • success: <b>{batchResult.n_success}</b> • failed:{" "}
                      <b>{batchResult.n_failed}</b>
                    </div>

                    <div className="mt-3 max-h-72 overflow-auto rounded-2xl border border-gray-200 bg-white">
                      <table className="w-full text-left text-sm">
                        <thead className="sticky top-0 bg-white">
                          <tr className="border-b border-gray-200">
                            <th className="p-2">row</th>
                            <th className="p-2">transaction_id</th>
                            <th className="p-2">bucket</th>
                            <th className="p-2">action</th>
                            <th className="p-2">risk_score</th>
                            <th className="p-2">status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {batchResult.results.slice(0, 200).map((r, i) => (
                            <tr key={i} className="border-b border-gray-100">
                              <td className="p-2">{String(r.row_index ?? i)}</td>
                              <td className="p-2 font-mono text-xs">{String(r.transaction_id ?? "-")}</td>
                              <td className="p-2">{String(r.risk_bucket ?? "-")}</td>
                              <td className="p-2">{String(r.recommended_action ?? "-")}</td>
                              <td className="p-2 font-mono text-xs">
                                {typeof r.risk_score === "number" ? r.risk_score.toFixed(6) : "-"}
                              </td>
                              <td className="p-2">{String(r.status ?? "-")}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    <div className="mt-2 text-xs text-gray-500">showing up to 200 rows in UI</div>
                  </div>
                )}
              </BentoCard>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

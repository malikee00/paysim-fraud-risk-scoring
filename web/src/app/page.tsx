"use client";

import { useMemo, useState } from "react";

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

export default function Home() {
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
    if (
      oldbalanceOrg < 0 ||
      newbalanceOrig < 0 ||
      oldbalanceDest < 0 ||
      newbalanceDest < 0
    )
      return true;
    return false;
  }, [step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]);

  async function handleScore() {
    setError(null);
    setResult(null);

    if (isInvalid) {
      setError("Input invalid: angka tidak boleh negatif.");
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
      setBatchError("Pilih file CSV dulu.");
      return;
    }

    if (!batchFile.name.toLowerCase().endsWith(".csv")) {
      setBatchError("File harus .csv");
      return;
    }

    setBatchLoading(true);

    try {
      const form = new FormData();
      form.append("file", batchFile);

      // Karena dropdown mode dihapus, kita fix ke raw
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

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900">
      <div className="mx-auto max-w-4xl px-4 py-10">
        <header className="mb-8">
          <h1 className="text-3xl font-bold">PaySim Fraud Risk</h1>
          <p className="mt-2 text-sm text-gray-600">
            Real-time scoring and upload sample data.
          </p>
          <p className="mt-1 text-xs text-gray-500">
            API Base: <span className="font-mono">{API_BASE}</span>
          </p>
        </header>

        {/* ===== Single score card ===== */}
        <section className="rounded-2xl bg-white p-6 shadow-sm">
          <h2 className="text-xl font-semibold">Score a Transaction</h2>

          <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <label className="text-sm font-medium">Step</label>
              <input
                type="number"
                className="mt-1 w-full rounded-lg border p-2"
                value={step}
                onChange={(e) => setStep(Number(e.target.value))}
              />
            </div>

            <div>
              <label className="text-sm font-medium">Type</label>
              <select
                className="mt-1 w-full rounded-lg border p-2"
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
              <label className="text-sm font-medium">Amount</label>
              <input
                type="number"
                className="mt-1 w-full rounded-lg border p-2"
                value={amount}
                onChange={(e) => setAmount(Number(e.target.value))}
              />
            </div>

            <div>
              <label className="text-sm font-medium">nameDest (optional)</label>
              <input
                className="mt-1 w-full rounded-lg border p-2 font-mono text-sm"
                value={nameDest}
                onChange={(e) => setNameDest(e.target.value)}
                placeholder="Cxxxx or Mxxxx"
              />
            </div>

            <div>
              <label className="text-sm font-medium">oldbalanceOrg</label>
              <input
                type="number"
                className="mt-1 w-full rounded-lg border p-2"
                value={oldbalanceOrg}
                onChange={(e) => setOldbalanceOrg(Number(e.target.value))}
              />
            </div>

            <div>
              <label className="text-sm font-medium">newbalanceOrig</label>
              <input
                type="number"
                className="mt-1 w-full rounded-lg border p-2"
                value={newbalanceOrig}
                onChange={(e) => setNewbalanceOrig(Number(e.target.value))}
              />
            </div>

            <div>
              <label className="text-sm font-medium">oldbalanceDest</label>
              <input
                type="number"
                className="mt-1 w-full rounded-lg border p-2"
                value={oldbalanceDest}
                onChange={(e) => setOldbalanceDest(Number(e.target.value))}
              />
            </div>

            <div>
              <label className="text-sm font-medium">newbalanceDest</label>
              <input
                type="number"
                className="mt-1 w-full rounded-lg border p-2"
                value={newbalanceDest}
                onChange={(e) => setNewbalanceDest(Number(e.target.value))}
              />
            </div>

            <div className="md:col-span-2">
              <label className="text-sm font-medium">nameOrig (optional)</label>
              <input
                className="mt-1 w-full rounded-lg border p-2 font-mono text-sm"
                value={nameOrig}
                onChange={(e) => setNameOrig(e.target.value)}
                placeholder="Cxxxx"
              />
            </div>
          </div>

          <div className="mt-5 flex items-center gap-3">
            <button
              onClick={handleScore}
              disabled={loading || isInvalid}
              className="rounded-xl bg-black px-4 py-2 text-white disabled:opacity-50"
            >
              {loading ? "Scoring..." : "Score Transaction"}
            </button>

            {isInvalid && (
              <span className="text-sm text-red-600">
                Input invalid: angka tidak boleh negatif.
              </span>
            )}
          </div>

          {error && (
            <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {result && (
            <div className="mt-6 rounded-2xl border bg-gray-50 p-5">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="text-sm text-gray-600">Risk Score</div>
                  <div className="text-2xl font-bold">{result.risk_score.toFixed(6)}</div>
                  <div className="mt-1 text-xs text-gray-500">
                    model: <span className="font-mono">{result.model_version ?? "-"}</span> •
                    thresholds:{" "}
                    <span className="font-mono">{result.thresholds_version ?? "-"}</span>
                  </div>
                </div>

                <div className="flex gap-2">
                  <span className="rounded-full bg-white px-3 py-1 text-sm font-medium shadow-sm">
                    bucket: <b>{result.risk_bucket}</b>
                  </span>
                  <span className="rounded-full bg-white px-3 py-1 text-sm font-medium shadow-sm">
                    action: <b>{result.recommended_action}</b>
                  </span>
                </div>
              </div>

              <div className="mt-4">
                <div className="h-3 w-full rounded-full bg-white">
                  <div
                    className="h-3 rounded-full bg-black"
                    style={{ width: scoreBar }}
                    title={fmtPct(result.risk_score)}
                  />
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  risk ≈ {fmtPct(result.risk_score)}
                </div>
              </div>
            </div>
          )}
        </section>

        {/* ===== Batch section ===== */}
        <section className="mt-8 rounded-2xl bg-white p-6 shadow-sm">
          <h2 className="text-xl font-semibold">Batch (CSV Upload)</h2>
          <p className="mt-1 text-sm text-gray-600">
            Please upload the provided sample data to trying the models.
          </p>

          <div className="mt-4">
            <a
              href="/demo_batch_raw.csv"
              download
              className="inline-flex rounded-xl border bg-blue-50 px-3 py-2 text-sm font-medium text-blue-700 hover:bg-blue-100"
            >
              Download Demo CSV
            </a>
          </div>

          <div className="mt-4 flex flex-col gap-3 md:flex-row md:items-center">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setBatchFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm"
            />
            <button
              onClick={handleBatchUpload}
              disabled={batchLoading}
              className="rounded-xl bg-black px-4 py-2 text-white disabled:opacity-50"
            >
              {batchLoading ? "Uploading..." : "Upload & Score"}
            </button>
          </div>

          {batchError && (
            <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {batchError}
            </div>
          )}

          {batchResult && (
            <div className="mt-6 rounded-2xl border bg-gray-50 p-5">
              <div className="text-sm text-gray-700">
                rows: <b>{batchResult.n_rows}</b> • success: <b>{batchResult.n_success}</b> •
                failed: <b>{batchResult.n_failed}</b>
              </div>

              <div className="mt-3 max-h-64 overflow-auto rounded-lg border bg-white">
                <table className="w-full text-left text-sm">
                  <thead className="sticky top-0 bg-white">
                    <tr className="border-b">
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
                      <tr key={i} className="border-b">
                        <td className="p-2">{String(r.row_index ?? i)}</td>
                        <td className="p-2 font-mono text-xs">
                          {String(r.transaction_id ?? "-")}
                        </td>
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

              <div className="mt-2 text-xs text-gray-500">
                showing up to 200 rows in UI
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

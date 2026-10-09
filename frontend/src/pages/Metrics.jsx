import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getMetrics } from "../api";

const EXPERIMENTS = ["augmented", "baseline"];
const METRIC_FIELDS = [
  ["Accuracy", "accuracy"],
  ["Precision", "precision"],
  ["Recall", "recall"],
  ["F1", "f1"],
  ["ROC-AUC", "roc_auc"],
];
const TONES = ["Light", "Medium", "Dark"];

export default function Metrics() {
  const [exp, setExp] = useState("augmented");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    getMetrics(exp)
      .then((d) => active && setData(d))
      .catch((err) => {
        const detail = err.response?.data?.detail || err.message;
        active && setError(`Failed to load metrics: ${detail}`);
      })
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [exp]);

  const toneData =
    data &&
    TONES.map((tone) => ({
      tone,
      accuracy: data.skin_tone?.[tone]?.accuracy ?? 0,
      count: data.skin_tone?.[tone]?.count ?? 0,
    }));

  return (
    <section className="page">
      <h1>Model Metrics</h1>

      <div className="controls">
        <label htmlFor="exp-select">Experiment:</label>
        <select
          id="exp-select"
          value={exp}
          onChange={(e) => setExp(e.target.value)}
        >
          {EXPERIMENTS.map((e) => (
            <option key={e} value={e}>{e}</option>
          ))}
        </select>
      </div>

      {loading && <p>Loading…</p>}
      {error && <p className="error" role="alert">{error}</p>}

      {data && !loading && (
        <>
          <div className="cards">
            {METRIC_FIELDS.map(([label, key]) => (
              <div className="card" key={key}>
                <span className="card-value">{data.overall[key].toFixed(3)}</span>
                <span className="card-label">{label}</span>
              </div>
            ))}
          </div>

          <h2>Confusion Matrix (DDI)</h2>
          <table className="confusion">
            <thead>
              <tr>
                <th></th>
                <th>Pred Benign</th>
                <th>Pred Malignant</th>
              </tr>
            </thead>
            <tbody>
              {["Actual Benign", "Actual Malignant"].map((label, i) => (
                <tr key={label}>
                  <th>{label}</th>
                  {data.confusion_matrix[i].map((v, j) => (
                    <td key={j}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          <h2>Per-Skin-Tone Accuracy</h2>
          <div className="chart">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={toneData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="tone" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Bar dataKey="accuracy" fill="#0072B2" name="Accuracy" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <p className="fairness">
            Light−Dark fairness gap:{" "}
            <strong>{data.fairness_gap.toFixed(3)}</strong>
          </p>
        </>
      )}
    </section>
  );
}

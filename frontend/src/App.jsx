import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const getJSON = async (url, opts) => {
  const res = await fetch(url, opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.detail || res.statusText);
  return body;
};

const fmtPrice = (v) => v?.toLocaleString(undefined, { maximumFractionDigits: 2 });
const fmtPct = (v) => (v == null ? "—" : `${(v * 100).toFixed(3)}%`);
const fmtNum = (v, d = 4) => (v == null ? "—" : Number(v).toFixed(d));

export default function App() {
  const [models, setModels] = useState([]);
  const [asset, setAsset] = useState("");
  const [run, setRun] = useState("");
  const [bars, setBars] = useState([]);
  const [news, setNews] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getJSON("/api/models")
      .then((m) => {
        setModels(m);
        if (m.length) setAsset(m[0].asset);
      })
      .catch((e) => setError(e.message));
  }, []);

  const assets = useMemo(() => [...new Set(models.map((m) => m.asset))], [models]);
  const assetRuns = models.filter((m) => m.asset === asset);
  const selected = models.find((m) => m.run === run);

  useEffect(() => {
    if (!asset) return;
    setRun(assetRuns[0]?.run || "");
    setPrediction(null);
    Promise.all([getJSON(`/api/market/${asset}?limit=500`), getJSON(`/api/news/${asset}?limit=30`)])
      .then(([b, n]) => {
        setBars(b);
        setNews(n);
      })
      .catch((e) => setError(e.message));
  }, [asset]);

  const predict = async () => {
    setLoading(true);
    setError("");
    try {
      setPrediction(
        await getJSON("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run }),
        })
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (!models.length && !error) return <main className="empty">Loading models…</main>;

  return (
    <main>
      <header>
        <h1>Multi-Factor Price Prediction</h1>
        <div className="controls">
          <label>
            Asset
            <select value={asset} onChange={(e) => setAsset(e.target.value)}>
              {assets.map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </label>
          <label>
            Model
            <select
              value={run}
              onChange={(e) => {
                setRun(e.target.value);
                setPrediction(null);
              }}
            >
              {assetRuns.map((m) => (
                <option key={m.run} value={m.run}>
                  {m.model_type.toUpperCase()} · {m.use_sentiment ? `price + sentiment (${m.sentiment_window})` : "price only"}
                </option>
              ))}
            </select>
          </label>
          <button onClick={predict} disabled={!run || loading}>
            {loading ? "Predicting…" : "Predict next hour"}
          </button>
        </div>
      </header>

      {error && <p className="error">{error}</p>}
      {!models.length && <p className="empty">No trained models found. Train one first (see README).</p>}

      <section className="grid">
        <div className="card chart">
          <h2>
            {asset} hourly close <span className="muted">last {bars.length} bars</span>
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={bars}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
              <XAxis dataKey="timestamp" tickFormatter={(t) => t.slice(5, 10)} minTickGap={40} stroke="var(--muted)" />
              <YAxis domain={["auto", "auto"]} tickFormatter={fmtPrice} width={80} stroke="var(--muted)" />
              <Tooltip formatter={fmtPrice} contentStyle={{ background: "var(--card)", border: "1px solid var(--grid)" }} />
              <Line type="monotone" dataKey="close" stroke="var(--accent)" dot={false} strokeWidth={1.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h2>Prediction</h2>
          {prediction ? (
            <div className="prediction">
              <div className={`direction ${prediction.direction.toLowerCase()}`}>
                {prediction.direction === "UP" ? "▲" : "▼"} {prediction.direction}
              </div>
              <dl>
                <dt>As of bar</dt>
                <dd>{prediction.as_of.replace("T", " ")} UTC</dd>
                <dt>Current price</dt>
                <dd>{fmtPrice(prediction.current_price)}</dd>
                <dt>Predicted price</dt>
                <dd>{fmtPrice(prediction.predicted_price)}</dd>
                <dt>Predicted return</dt>
                <dd>{fmtPct(prediction.predicted_return)}</dd>
                <dt>Confidence</dt>
                <dd>{fmtNum(prediction.confidence, 2)}</dd>
                {prediction.sentiment_score != null && (
                  <>
                    <dt>Sentiment (window)</dt>
                    <dd>{fmtNum(prediction.sentiment_score, 3)}</dd>
                  </>
                )}
              </dl>
              <p className="note" title={prediction.confidence_method}>
                Confidence is a relative uncertainty score (Monte Carlo dropout), not a probability of being right.
              </p>
            </div>
          ) : (
            <p className="muted">Pick a model and press “Predict next hour”.</p>
          )}
          <p className="note">Not investment advice.</p>
        </div>

        {selected && (
          <div className="card wide">
            <h2>
              Test-set metrics <span className="muted">{selected.n_test} held-out bars</span>
            </h2>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                  <th>R²</th>
                  <th>Direction acc.</th>
                  <th>F1 up</th>
                  <th>F1 down</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(selected.metrics).map(([name, m]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{fmtNum(m.mae, 5)}</td>
                    <td>{fmtNum(m.rmse, 5)}</td>
                    <td>{fmtNum(m.r2, 4)}</td>
                    <td>{(m.directional_accuracy * 100).toFixed(1)}%</td>
                    <td>{fmtNum(m.f1_up, 3)}</td>
                    <td>{fmtNum(m.f1_down, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="plots">
              {selected.plots.map((p) => (
                <a key={p} href={p} target="_blank" rel="noreferrer">
                  <img src={p} alt={p.split("/").pop()} />
                </a>
              ))}
            </div>
          </div>
        )}

        <div className="card wide">
          <h2>Latest headlines</h2>
          <ul className="news">
            {news.map((n) => (
              <li key={n.timestamp + n.text}>
                <time>{n.timestamp.replace("T", " ").slice(0, 16)}</time> {n.text}
              </li>
            ))}
          </ul>
        </div>
      </section>
    </main>
  );
}

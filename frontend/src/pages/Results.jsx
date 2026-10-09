import { Link, useNavigate } from "react-router-dom";
import { useResult } from "../ResultContext";

export default function Results() {
  const { result } = useResult();
  const navigate = useNavigate();

  if (!result) {
    return (
      <section className="page">
        <h1>Prediction Result</h1>
        <p className="empty">
          No analysis yet. Head to the{" "}
          <Link to="/">Upload</Link> page to analyze a lesion image.
        </p>
      </section>
    );
  }

  const { prediction, probability, model, originalUrl, overlayUrl } = result;
  const isMalignant = prediction === "Malignant";

  return (
    <section className="page">
      <h1>Prediction Result</h1>

      <div className="result-summary">
        <span className={`badge ${isMalignant ? "malignant" : "benign"}`}>
          {prediction}
        </span>
        <p>
          Probability of malignant: <strong>{(probability * 100).toFixed(1)}%</strong>
        </p>
        <p className="muted">Model: {model}</p>
      </div>

      <div className="image-pair">
        <figure>
          <img src={originalUrl} alt="Original lesion" />
          <figcaption>Original</figcaption>
        </figure>
        <figure>
          <img src={overlayUrl} alt="Grad-CAM heatmap overlay" />
          <figcaption>Grad-CAM Overlay</figcaption>
        </figure>
      </div>

      <button onClick={() => navigate("/")}>Analyze another</button>
    </section>
  );
}

import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { predict, heatmap } from "../api";
import { useResult } from "../ResultContext";

const MODELS = ["augmented", "baseline"];

export default function Upload() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [model, setModel] = useState("augmented");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);
  const navigate = useNavigate();
  const { setResult } = useResult();

  function selectFile(selected) {
    if (!selected) return;
    setError(null);
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  }

  function onDrop(event) {
    event.preventDefault();
    selectFile(event.dataTransfer.files?.[0]);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const [prediction, overlayUrl] = await Promise.all([
        predict(file, model),
        heatmap(file, model),
      ]);
      setResult({ ...prediction, originalUrl: previewUrl, overlayUrl });
      navigate("/results");
    } catch (err) {
      const detail = err.response?.data?.detail || err.message;
      setError(`Analysis failed: ${detail}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="page">
      <h1>Upload a Lesion Image</h1>
      <div
        className="dropzone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload image: drag and drop or click to browse"
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Selected lesion preview" className="preview" />
        ) : (
          <p>Drag &amp; drop an image here, or click to browse (JPEG/PNG).</p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          hidden
          onChange={(e) => selectFile(e.target.files?.[0])}
        />
      </div>

      <div className="controls">
        <label htmlFor="model-select">Model:</label>
        <select
          id="model-select"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={loading}
        >
          {MODELS.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <button onClick={analyze} disabled={!file || loading}>
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </div>

      {error && <p className="error" role="alert">{error}</p>}
    </section>
  );
}

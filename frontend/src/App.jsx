import { Link, Navigate, Route, Routes } from "react-router-dom";
import Upload from "./pages/Upload";
import Results from "./pages/Results";
import Metrics from "./pages/Metrics";

export default function App() {
  return (
    <div className="app">
      <header className="nav">
        <span className="brand">Melanoma SOC</span>
        <nav>
          <Link to="/">Upload</Link>
          <Link to="/results">Results</Link>
          <Link to="/metrics">Metrics</Link>
        </nav>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/results" element={<Results />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="disclaimer">
        Research prototype only — not a medical device and not for diagnostic use.
      </footer>
    </div>
  );
}
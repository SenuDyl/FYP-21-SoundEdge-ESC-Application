import { useMemo, useState } from "react";

import { useRemoteList } from "./hooks/useRemoteList";
import SelectField from "./components/SelectField";
import SamplePlayer from "./components/SamplePlayer";
import ResultCard from "./components/ResultCard";
import ExplanationsCard from "./components/ExplanationsCard";
import VisualizationsPanel from "./components/VisualizationsPanel";

const BACKEND_URL = "http://localhost:8000";

export default function App() {
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [result, setResult] = useState(null);

  const {
    items: samples,
    selected: selectedSample,
    setSelected: setSelectedSample,
    loading: samplesLoading,
  } = useRemoteList({
    url: `${BACKEND_URL}/samples`,
    keyName: "samples",
    errorMessage: "Failed to load samples.",
  });

  const {
    items: models,
    selected: selectedModel,
    setSelected: setSelectedModel,
    loading: modelsLoading,
  } = useRemoteList({
    url: `${BACKEND_URL}/models`,
    keyName: "models",
    errorMessage: "Failed to load models.",
  });

  const audioUrl = useMemo(() => {
    if (!selectedSample) return null;
    return `${BACKEND_URL}/samples/${encodeURIComponent(selectedSample)}`;
  }, [selectedSample]);

  function clearAll() {
    setResult(null);
    setErr("");
    setSelectedSample(samples?.[0] || "");
    setSelectedModel(models?.[0] || "");
  }

  function playSegment(segment) {
    const audio = new Audio(`data:audio/wav;base64,${segment.audio_base64}`);
    audio.currentTime = 0;
    audio.play();
  }

  async function onPredict() {
    setErr("");
    setResult(null);

    if (!selectedSample) {
      setErr("Please select an audio sample.");
      return;
    }

    if (!selectedModel) {
      setErr("Please select a model.");
      return;
    }

    setLoading(true);
    try {
      const form = new FormData();
      form.append("sample_name", selectedSample);
      form.append("model_name", selectedModel);

      const res = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Prediction failed.");
      }

      const data = await res.json();
      setResult(data);
    } catch (e) {
      setErr(e?.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <aside className="left">
        <div className="leftInner">
          <div className="card">
            <div className="headerTitle">Environmental Sound Classifier</div>
            <div className="headerSub">
              Using CNN-PSK Model and Explainable AI (XAI) for Enhanced Audio Recognition
            </div>
          </div>

          <div className="card">
            <div className="sectionTitle">Input</div>

            <SelectField
              label="Model"
              value={selectedModel}
              onChange={setSelectedModel}
              items={models}
              loading={modelsLoading}
              disabled={loading}
              placeholderLoading="Loading models..."
              placeholderEmpty="No models found"
            />

            <SelectField
              label="Audio Sample"
              value={selectedSample}
              onChange={setSelectedSample}
              items={samples}
              loading={samplesLoading}
              disabled={loading}
              placeholderLoading="Loading samples..."
              placeholderEmpty="No samples found"
            />

            <SamplePlayer selectedSample={selectedSample} audioUrl={audioUrl} />

            <div style={{ marginTop: 12, display: "flex", gap: 10 }}>
              <button onClick={clearAll} disabled={loading} className="btn">
                Clear
              </button>

              <button
                onClick={onPredict}
                disabled={loading || !selectedSample || !selectedModel}
                className="btn btnPrimary"
              >
                {loading ? "Classifying..." : "Classify"}
              </button>
            </div>

            {err && <div className="errorBox">{err}</div>}
          </div>

          <ResultCard result={result} />
          <ExplanationsCard result={result} onPlaySegment={playSegment} />
        </div>
      </aside>

      <VisualizationsPanel result={result} />
    </div>
  );
}

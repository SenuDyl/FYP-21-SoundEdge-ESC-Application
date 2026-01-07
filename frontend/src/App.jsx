import { useEffect, useMemo, useState } from "react";

const BACKEND_URL = "http://localhost:8000";

export default function App() {
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState("");
  const [samplesLoading, setSamplesLoading] = useState(false);

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [modelsLoading, setModelsLoading] = useState(false);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [result, setResult] = useState(null);

  // Load sample list
  useEffect(() => {
    async function loadSamples() {
      setSamplesLoading(true);
      try {
        const res = await fetch(`${BACKEND_URL}/samples`);
        if (!res.ok) throw new Error("Failed to load samples.");
        const data = await res.json();
        const list = Array.isArray(data.samples) ? data.samples : [];
        setSamples(list);
        if (list.length > 0) setSelectedSample(list[0]);
      } catch (e) {
        setErr(e?.message || "Failed to load samples.");
      } finally {
        setSamplesLoading(false);
      }
    }
    loadSamples();
  }, []);

  // Load model list
  useEffect(() => {
    async function loadModels() {
      setModelsLoading(true);
      try {
        const res = await fetch(`${BACKEND_URL}/models`);
        if (!res.ok) throw new Error("Failed to load models.");
        const data = await res.json();
        const list = Array.isArray(data.models) ? data.models : [];
        setModels(list);
        if (list.length > 0) setSelectedModel(list[0]);
      } catch (e) {
        setErr(e?.message || "Failed to load models.");
      } finally {
        setModelsLoading(false);
      }
    }
    loadModels();
  }, []);

  // URL for playing selected sample from backend
  const audioUrl = useMemo(() => {
    if (!selectedSample) return null;
    return `${BACKEND_URL}/samples/${encodeURIComponent(selectedSample)}`;
  }, [selectedSample]);

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

  function onClear() {
    setSelectedSample(samples?.[0] || "");
    setSelectedModel(models?.[0] || "");
    setResult(null);
    setErr("");
  }

  function playSegment(segment) {
    const audio = new Audio(`data:audio/wav;base64,${segment.audio_base64}`);
    audio.currentTime = 0;
    audio.play();
  }

  const segments =
    result?.time_explanation?.finalized_segments &&
    Array.isArray(result.time_explanation.finalized_segments)
      ? result.time_explanation.finalized_segments
      : [];

  const images = [
    result?.spectrogram
      ? {
          title: "Mel Spectrogram",
          src: result.spectrogram,
          subtitle: "Time–frequency representation of the input audio signal",
        }
      : null,
    result?.gradcam_heatmap
      ? {
          title: "Grad-CAM Heatmap",
          src: result.gradcam_heatmap,
          subtitle: "Visualization of regions where the model focuses most",
        }
      : null,
    result?.spectrogram_bboxes
      ? {
          title: "Mel Spectrogram with Bounding Boxes",
          src: result.spectrogram_bboxes,
          subtitle:
            "Visualization of the spectrogram with annotated regions of interest",
        }
      : null,
    result?.gradcam_heatmap_bboxes
      ? {
          title: "Grad-CAM Heatmap with Bounding Boxes",
          src: result.gradcam_heatmap_bboxes,
          subtitle:
            "Focused areas overlaid using bounding boxes for better understanding",
        }
      : null,
  ].filter(Boolean);

  return (
    <div className="app">
      {/* Left panel */}
      <aside className="left">
        <div className="leftInner">
          <div className="card">
            <div className="headerTitle">Environmental Sound Classifier</div>
            <div className="headerSub">
              Using CNN-PSK Model and Explainable AI (XAI) for Enhanced Audio
              Recognition
            </div>
          </div>

          <div className="card">
            <div className="sectionTitle">Input</div>

            {/* Model dropdown */}
            <div style={{ marginTop: 10, marginBottom: 10 }}>
              <div className="muted" style={{ marginBottom: 6 }}>
                Model
              </div>
              <select
                className="select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={modelsLoading || models.length === 0 || loading}
              >
                {models.length === 0 ? (
                  <option value="">
                    {modelsLoading ? "Loading models..." : "No models found"}
                  </option>
                ) : (
                  models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))
                )}
              </select>
            </div>

            {/* Sample dropdown */}
            <div style={{ marginTop: 10, marginBottom: 10 }}>
              <div className="muted" style={{ marginBottom: 6 }}>
                Audio Sample
              </div>
              <select
                className="select"
                value={selectedSample}
                onChange={(e) => setSelectedSample(e.target.value)}
                disabled={samplesLoading || samples.length === 0 || loading}
              >
                {samples.length === 0 ? (
                  <option value="">
                    {samplesLoading ? "Loading samples..." : "No samples found"}
                  </option>
                ) : (
                  samples.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))
                )}
              </select>
            </div>

            {/* Sample player */}
            {selectedSample && audioUrl && (
              <div style={{ marginTop: 12 }}>
                <div className="muted">
                  Selected: <b>{selectedSample}</b>
                </div>
                <audio
                  controls
                  src={audioUrl}
                  style={{ width: "100%", marginTop: 12, padding: 0 }}
                />
              </div>
            )}

            {/* Actions */}
            <div style={{ marginTop: 12, display: "flex", gap: 10 }}>
              <button onClick={onClear} disabled={loading} className="btn">
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

          {result && (
            <div className="card">
              <div className="sectionTitle">Result</div>

              <div className="resultRow">
                <div>
                  <div className="muted">Predicted</div>
                  <div className="bigText">{result.label}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div className="muted">Confidence</div>
                  <div className="bigText">
                    {result.confidence != null
                      ? `${(result.confidence * 100).toFixed(2)}%`
                      : "-"}
                  </div>
                </div>
              </div>

              {Array.isArray(result.topk) && result.topk.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div className="muted">Top predictions</div>
                  <ul style={{ marginTop: 6, paddingLeft: 18 }}>
                    {result.topk.map((x, i) => (
                      <li key={i}>
                        {x.label}{" "}
                        {x.confidence != null
                          ? `(${(x.confidence * 100).toFixed(2)}%)`
                          : ""}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {result && (
            <div className="card">
              <div className="sectionTitle">Explanations</div>

              {result.frequency_explanation && (
                <div style={{ marginTop: 10 }}>
                  <div className="muted">Frequency</div>
                  <div className="textBox">{result.frequency_explanation}</div>
                </div>
              )}

              {result.time_explanation && (
                <div style={{ marginTop: 12 }}>
                  <div className="muted">Temporal</div>
                  <div className="textBox">
                    Temporal characteristics:{" "}
                    <b>{result.time_explanation.temporal_category}</b>
                  </div>
                </div>
              )}

              {segments.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div className="muted">
                    Model focused on sound segments between:
                  </div>
                  <div style={{ marginTop: 8, display: "grid", gap: 8 }}>
                    {segments.map((seg, idx) => (
                      <div key={idx} className="segmentRow">
                        <div style={{ fontSize: 13, color: "#222" }}>
                          <b>Segment {seg.segment_index}</b>
                          <div className="muted">
                            {seg.start_sec.toFixed(2)} – {seg.end_sec.toFixed(2)}{" "}
                            s
                          </div>
                        </div>
                        <button
                          onClick={() => playSegment(seg)}
                          className="btn btnSmall"
                        >
                          ▶ Play
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="muted" style={{ marginTop: 14 }}>
                Tip: Try a few different samples and compare how the explanations
                change between models.
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Right panel */}
      <main className="right">
        <div className="rightInner">
          {!result && (
            <div className="card">
              <div style={{ fontSize: 18, fontWeight: 700, color: "#222" }}>
                Select an audio sample and click Classify
              </div>
              <div style={{ marginTop: 6 }} className="muted">
                Spectrograms and Grad-CAM visualizations will appear here.
              </div>
            </div>
          )}

          {result && (
            <div>
              <div className="rightHeader">
                <div style={{ fontSize: 16, fontWeight: 700, color: "#222" }}>
                  Visual Explanations
                </div>
              </div>

              <div className="grid">
                {images.map((img, i) => (
                  <div key={i} className="card">
                    <div className="imageTitle">{img.title}</div>
                    <div className="imageSubtitle">{img.subtitle}</div>
                    <img alt={img.title} src={img.src} className="img" />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

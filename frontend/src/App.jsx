import { useMemo, useState } from "react";

const BACKEND_URL = "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  // eslint-disable-next-line no-unused-vars
  const [explain, setExplain] = useState(true);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [result, setResult] = useState(null);

  const audioUrl = useMemo(() => {
    if (!file) return null;
    return URL.createObjectURL(file);
  }, [file]);

  async function onPredict() {
    setErr("");
    setResult(null);

    if (!file) {
      setErr("Please select an audio file.");
      return;
    }

    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(
        `${BACKEND_URL}/predict?explain=${explain ? "true" : "false"}`,
        { method: "POST", body: form }
      );

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
    setFile(null);
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
    result?.spectrogram ? { title: "Mel Spectrogram", src: result.spectrogram } : null,
    result?.gradcam_heatmap ? { title: "Grad-CAM Heatmap", src: result.gradcam_heatmap } : null,
    result?.spectrogram_bboxes ? { title: "Mel Spectrogram + BBoxes", src: result.spectrogram_bboxes } : null,
    result?.gradcam_heatmap_bboxes ? { title: "Grad-CAM Heatmap + BBoxes", src: result.gradcam_heatmap_bboxes } : null,
  ].filter(Boolean);

  return (
    <div className="app">
      {/* Left panel */}
      <aside className="left">
        <div className="leftInner">
          <div className="card">
            <div className="headerTitle">ESC-10 Sound Classifier</div>
            <div className="headerSub">CNN-PSK + XAI outputs</div>
          </div>

          <div className="card">
            <div className="sectionTitle">Input</div>

            <div className="row">
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
              <button
                onClick={onClear}
                disabled={loading && !file}
                className="btn"
              >
                Clear
              </button>
            </div>

            {file && (
              <div style={{ marginTop: 12 }}>
                <div className="muted">
                  Selected: <b>{file.name}</b> ({Math.round(file.size / 1024)} KB)
                </div>
                <audio controls src={audioUrl} style={{ width: "100%", marginTop: 12, padding: 0 }} />
              </div>
            )}

            <div style={{ marginTop: 12, display: "flex", gap: 10 }}>
              <button
                onClick={onPredict}
                disabled={loading || !file}
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
                    {result.confidence != null ? `${(result.confidence * 100).toFixed(2)}%` : "-"}
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
                        {x.confidence != null ? `(${(x.confidence * 100).toFixed(2)}%)` : ""}
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
                    Temporal characteristics: <b>{result.time_explanation.temporal_category}</b>
                  </div>
                </div>
              )}

              {segments.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div className="muted">Model focused on sound segments between:</div>
                  <div style={{ marginTop: 8, display: "grid", gap: 8 }}>
                    {segments.map((seg, idx) => (
                      <div key={idx} className="segmentRow">
                        <div style={{ fontSize: 13, color: "#222" }}>
                          <b>Segment {seg.segment_index}</b>
                          <div className="muted">
                            {seg.start_sec.toFixed(2)} – {seg.end_sec.toFixed(2)} s
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
                Tip: WAV files usually work best. Keep clips close to the sound source.
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
                Upload an audio file and click Classify
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

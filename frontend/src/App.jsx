import { useMemo, useState } from "react";

const BACKEND_URL = "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
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

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h2 style={{ marginTop: 0 }}>ESC-10 Sound Classifier (CNN-PSK)</h2>

        <div style={styles.row}>
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <button onClick={onClear} disabled={loading && !file}>
            Clear
          </button>
        </div>

        {file && (
          <div style={{ marginTop: 12 }}>
            <div style={styles.muted}>
              Selected: <b>{file.name}</b> ({Math.round(file.size / 1024)} KB)
            </div>

            <audio
              controls
              src={audioUrl}
              style={{ width: "100%", marginTop: 8 }}
            />
          </div>
        )}

        <div style={{ marginTop: 12 }}>
          <button
            onClick={onPredict}
            disabled={loading || !file}
            style={styles.primaryBtn}
          >
            {loading ? "Classifying..." : "Classify"}
          </button>
        </div>

        {err && <p style={styles.error}>{err}</p>}

        {result && (
          <div style={{ marginTop: 18 }}>
            <h3 style={{ marginBottom: 8 }}>Result</h3>

            <div style={styles.resultRow}>
              <div>
                <div style={styles.muted}>Predicted label</div>
                <div style={styles.bigText}>{result.label}</div>
              </div>

              <div>
                <div style={styles.muted}>Confidence</div>
                <div style={styles.bigText}>
                  {result.confidence != null
                    ? `${(result.confidence * 100).toFixed(2)}%`
                    : "-"}
                </div>
              </div>
            </div>

            {Array.isArray(result.topk) && result.topk.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={styles.muted}>Top predictions</div>
                <ul style={{ marginTop: 6 }}>
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

            {result.spectrogram && (
              <div style={{ marginTop: 14 }}>
                <div style={styles.muted}>Explanation</div>
                <img
                  alt="Grad-CAM explanation"
                  src={result.spectrogram}
                  style={styles.img}
                />
              </div>
            )}
            {result.gradcam_heatmap && (
              <div style={{ marginTop: 14 }}>
                {/* <div style={styles.muted}>Explanation</div> */}
                <img
                  alt="Grad-CAM explanation"
                  src={result.gradcam_heatmap}
                  style={styles.img}
                />
              </div>
            )}
            {result.spectrogram_bboxes && (
              <div style={{ marginTop: 14 }}>
                {/* <div style={styles.muted}>Explanation</div> */}
                <img
                  alt="Grad-CAM explanation"
                  src={result.spectrogram_bboxes}
                  style={styles.img}
                />
              </div>
            )}
            {result.gradcam_heatmap_bboxes && (
              <div style={{ marginTop: 14 }}>
                {/* <div style={styles.muted}>Explanation</div> */}
                <img
                  alt="Grad-CAM explanation"
                  src={result.gradcam_heatmap_bboxes}
                  style={styles.img}
                />
              </div>
            )}
            {result.frequency_explanation && (
              <div style={{ marginTop: 14 }}>
                <div style={styles.muted}>Frequency Explanation</div>
                <div style={{ padding: 10, borderRadius: 8, color: "#222", background: "#f0f0f0", marginTop: 6 }}>
                  {result.frequency_explanation}
                </div>
              </div>
            )}

            {result.time_explanation && (
              <div style={{ marginTop: 14 }}>
                <div style={styles.muted}>
                  Temporal Explanation
                </div>
                <div style={{ padding: 10, borderRadius: 8, color: "#222", background: "#f0f0f0", marginTop: 6 }}>
                  Temporal characteristics: <b>{result.time_explanation.temporal_category}</b>
                </div>

                {Array.isArray(result.time_explanation.finalized_segments) &&
                  result.time_explanation.finalized_segments.length > 0 && (
                    <div style={{ marginTop: 14 }}>
                      <div style={{ color: "#333", lineHeight: 1.6 }}>
                        The model focused on sound events between:
                        <ul style={{ marginTop: 4, paddingLeft: 20 }}>
                          {result.time_explanation.finalized_segments.map((seg, idx) => (
                            <li key={idx}>
                              {`Segment ${seg.segment_index}: ${seg.start_sec.toFixed(2)} – ${seg.end_sec.toFixed(2)} s`}
                              <button
                                onClick={() => playSegment(seg)}
                                style={{ marginLeft: 8, ...styles.smallBtn }}
                              >
                                ▶ Play
                              </button>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
              </div>
            )}

          </div>
        )}

        <p style={{ ...styles.muted, marginTop: 18 }}>
          Tip: WAV files usually work best. Keep clips close to the sound
          source.
        </p>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    display: "flex",
    justifyContent: "center",
    padding: 24,
    background: "#f6f7fb",
    fontFamily: "Arial, sans-serif",
  },
  card: {
    width: "100%",
    maxWidth: 720,
    background: "white",
    borderRadius: 12,
    padding: 18,
    boxShadow: "0 6px 18px rgba(0,0,0,0.08)",
  },
  row: {
    display: "flex",
    gap: 10,
    alignItems: "center",
    justifyContent: "space-between",
    flexWrap: "wrap",
  },
  primaryBtn: {
    padding: "10px 14px",
    borderRadius: 10,
    border: "none",
    cursor: "pointer",
    background: "#2f6fed",
    color: "white",
    fontWeight: 600,
  },
  muted: { color: "#555", fontSize: 13 },
  error: { color: "#b00020", marginTop: 10 },
  resultRow: {
    display: "flex",
    gap: 18,
    alignItems: "center",
    justifyContent: "space-between",
    flexWrap: "wrap",
    background: "#f3f5fb",
    padding: 12,
    borderRadius: 10,
  },
  bigText: { fontSize: 22, fontWeight: 700, color: "#222" },
  img: {
    width: "100%",
    borderRadius: 10,
    border: "1px solid #e5e5e5",
    marginTop: 8,
  },
};

function SegmentList({ segments, onPlay }) {
  if (!segments || segments.length === 0) return null;

  return (
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
            <button onClick={() => onPlay(seg)} className="btn btnSmall">
              ▶ Play
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ExplanationsCard({ result, onPlaySegment }) {
  if (!result) return null;

  const segments =
    Array.isArray(result?.time_explanation?.finalized_segments)
      ? result.time_explanation.finalized_segments
      : [];

  return (
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

      <SegmentList segments={segments} onPlay={onPlaySegment} />

      <div className="muted" style={{ marginTop: 14 }}>
        Tip: Try a few different samples and compare how the explanations change between models.
      </div>
    </div>
  );
}

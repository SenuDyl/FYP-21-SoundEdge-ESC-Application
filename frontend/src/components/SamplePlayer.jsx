export default function SamplePlayer({ selectedSample, audioUrl }) {
  if (!selectedSample || !audioUrl) return null;

  return (
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
  );
}

export default function ResultCard({ result }) {
  if (!result) return null;

  return (
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
  );
}

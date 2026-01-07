export default function VisualizationsPanel({ result }) {
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
          subtitle: "Visualization of the spectrogram with annotated regions of interest",
        }
      : null,
    result?.gradcam_heatmap_bboxes
      ? {
          title: "Grad-CAM Heatmap with Bounding Boxes",
          src: result.gradcam_heatmap_bboxes,
          subtitle: "Focused areas overlaid using bounding boxes for better understanding",
        }
      : null,
  ].filter(Boolean);

  return (
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
  );
}

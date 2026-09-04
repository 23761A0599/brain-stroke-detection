import React, { useState } from "react";
import { predictImage } from "./services/api";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setFileName(file.name);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError("");
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setFileName("");
    setPreview(null);
    setResult(null);
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError("Please upload an MRI image scan first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const data = await predictImage(selectedFile);
      setResult(data);
    } catch (err) {
      setError(
        err.response?.data?.detail || err.message || "Failed to process scan on server."
      );
      console.error("Prediction Error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to safely resolve base64 strings or standard URLs
  const formatImageSource = (imgData) => {
    if (!imgData) return preview;
    if (imgData.startsWith("data:image/") || imgData.startsWith("http://") || imgData.startsWith("https://")) {
      return imgData;
    }
    return `data:image/png;base64,${imgData}`;
  };

  // Logic to handle prediction mapping and numeric parse guarding
  const rawConfidence = typeof result?.confidence === "number" 
    ? result.confidence 
    : parseFloat(result?.confidence || 0);

  const isHemorrhagic =
    result?.prediction === "Hemorrhagic" ||
    result?.prediction === "NonHemorrhagic" ||
    result?.prediction === "Normal" ||
    result?.prediction === 0;

  const hemorrhagicProb = isHemorrhagic ? rawConfidence : 1 - rawConfidence;
  const nonHemorrhagicProb = isHemorrhagic ? 1 - rawConfidence : rawConfidence;

  const displayHemorrhagicPct = (hemorrhagicProb * (hemorrhagicProb > 1 ? 1 : 100)).toFixed(2);
  const displayNonHemorrhagicPct = (nonHemorrhagicProb * (nonHemorrhagicProb > 1 ? 1 : 100)).toFixed(2);
  const displayMainConfidence = (rawConfidence * (rawConfidence > 1 ? 1 : 100)).toFixed(2);

  // Fallback check for Pro Grad-CAM property name variations
  const proGradCamImage = result?.gradcam_pro || result?.pro_gradcam || result?.proGradcam;

  return (
    <div style={styles.appWrapper}>
      {/* Top Header */}
      <header style={styles.header}>
        <div style={styles.headerTag}>SYSTEM ACTIVE</div>
        <h1 style={styles.headerTitle}>BRAIN HEMORRHAGE DETECTION SYSTEM</h1>
        <p style={styles.headerSubtitle}>
          Deep Learning-based MRI Analysis using <strong>EfficientNet-B0</strong> • 
          FastAPI • React • <strong>Grad-CAM</strong> • <strong>LIME Explainability</strong>
        </p>
      </header>

      {/* Main Page Layout Container */}
      <main style={styles.mainContainer}>
        {/* 1. Project Information Card */}
        <section style={styles.card}>
          <h2 style={styles.cardTitleBlue}>PROJECT INFORMATION</h2>
          <h3 style={styles.projectHeading}>
            A Deep Neural Network Approach for Hemorrhagic Stroke Identification in MRI Scans
          </h3>
          <div style={styles.infoGrid}>
            <div style={styles.infoBox}>
              <p style={styles.infoLabel}>👨‍🏫 UNDER THE GUIDANCE OF</p>
              <p style={styles.infoBoldValue}>Mr. N. Srinivasa Rao</p>
              <p style={styles.infoSubText}>Sr. Assistant Professor</p>

              <div style={{ marginTop: "20px" }}>
                <p style={styles.infoLabel}>🏫 DEPARTMENT</p>
                <p style={styles.infoBoldValue}>Computer Science and Engineering</p>
              </div>
            </div>

            <div style={styles.infoBox}>
              <p style={styles.infoLabel}>👨‍🎓 PRESENTED BY</p>
              <p style={styles.infoBoldValue}>K. Subba Rao &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;23761A0599</p>
              <p style={styles.infoBoldValue}>D. Kowshik Reddy 23761A0584</p>
              <p style={styles.infoBoldValue}>Sh. Thasleem &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;23761A05C2</p>

              <div style={{ marginTop: "20px" }}>
                <p style={styles.infoLabel}>🎓 COLLEGE</p>
                <p style={styles.infoBoldValue}>Lakireddy Bali Reddy College of Engineering</p>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Upload, Prediction & Class Probabilities Row */}
        <div style={styles.dashboardGrid}>
          {/* Upload Box */}
          <div style={styles.smallCard}>
            <h3 style={styles.cardTitleBlue}>UPLOAD MRI SCAN</h3>
            <form onSubmit={handleSubmit} style={styles.uploadForm}>
              <div style={styles.dashedUploadBox}>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  style={{ width: "100%", cursor: "pointer", fontWeight: "600", fontSize: "13px" }}
                />
              </div>

              {fileName && (
                <div style={styles.fileNameBadge}>
                  📄 <strong>{fileName}</strong>
                </div>
              )}

              {preview && (
                <div style={styles.previewContainer}>
                  <img src={preview} alt="Scan Preview" style={styles.previewImage} />
                </div>
              )}

              <div style={{ width: "100%", display: "flex", gap: "10px" }}>
                <button
                  type="submit"
                  disabled={loading || !selectedFile}
                  style={{
                    ...styles.predictBtn,
                    opacity: loading || !selectedFile ? 0.5 : 1,
                    cursor: loading || !selectedFile ? "not-allowed" : "pointer"
                  }}
                >
                  {loading ? "PROCESSING SCAN..." : "PREDICT RESULT"}
                </button>

                {(selectedFile || result) && (
                  <button
                    type="button"
                    onClick={handleReset}
                    style={styles.resetBtn}
                  >
                    RESET
                  </button>
                )}
              </div>
            </form>
          </div>

          {/* Prediction Result Box */}
          <div style={styles.smallCard}>
            <h3 style={styles.cardTitleBlue}>PREDICTION RESULT</h3>
            {result ? (
              <div style={styles.resultContainer}>
                <div style={styles.resultRow}>
                  <span style={styles.rowLabel}>DIAGNOSIS</span>
                  <span
                    style={{
                      ...styles.statusTag,
                      backgroundColor: isHemorrhagic ? "#fef2f2" : "#f0fdf4",
                      color: isHemorrhagic ? "#b91c1c" : "#15803d",
                      borderColor: isHemorrhagic ? "#fca5a5" : "#86efac",
                    }}
                  >
                    {isHemorrhagic ? "⚠️ HEMORRHAGIC" : "✅ NON-HEMORRHAGIC"}
                  </span>
                </div>

                <div style={styles.resultRow}>
                  <span style={styles.rowLabel}>CONFIDENCE</span>
                  <strong style={styles.confidenceText}>
                    {result.confidence_percentage || `${displayMainConfidence}%`}
                  </strong>
                </div>
              </div>
            ) : (
              <div style={styles.emptyStateBox}>
                <p style={{ margin: 0, color: "#94a3b8", fontSize: "14px", fontWeight: "600" }}>
                  Awaiting scan upload...
                </p>
              </div>
            )}
          </div>

          {/* Class Probabilities Box */}
          <div style={styles.smallCard}>
            <h3 style={styles.cardTitleBlue}>CLASS PROBABILITIES</h3>
            {result ? (
              <div style={styles.probContainer}>
                <div style={styles.probHeader}>
                  <strong style={{ color: "#dc2626" }}>HEMORRHAGIC</strong>
                  <strong>{displayHemorrhagicPct}%</strong>
                </div>
                <div style={styles.progressTrack}>
                  <div
                    style={{
                      ...styles.progressFill,
                      width: `${displayHemorrhagicPct}%`,
                      backgroundColor: "#ef4444",
                    }}
                  />
                </div>

                <div style={{ ...styles.probHeader, marginTop: "24px" }}>
                  <strong style={{ color: "#2563eb" }}>NON-HEMORRHAGIC</strong>
                  <strong>{displayNonHemorrhagicPct}%</strong>
                </div>
                <div style={styles.progressTrack}>
                  <div
                    style={{
                      ...styles.progressFill,
                      width: `${displayNonHemorrhagicPct}%`,
                      backgroundColor: "#2563eb",
                    }}
                  />
                </div>
              </div>
            ) : (
              <div style={styles.emptyStateBox}>
                <p style={{ margin: 0, color: "#94a3b8", fontSize: "14px", fontWeight: "600" }}>
                  Probabilities will appear here.
                </p>
              </div>
            )}
          </div>
        </div>

        {error && <div style={styles.errorCard}><strong>ERROR:</strong> {error}</div>}

        {/* 3. Medical Interpretation */}
        {result && (
          <section style={styles.card}>
            <h2 style={styles.cardTitleBlue}>MEDICAL INTERPRETATION</h2>
            <div style={styles.interpretationText}>
              <p>
                <strong>DIAGNOSIS STATUS:</strong>{" "}
                <strong style={{ color: isHemorrhagic ? "#dc2626" : "#16a34a" }}>
                  {isHemorrhagic ? "HEMORRHAGIC" : "NON-HEMORRHAGIC"}
                </strong>
              </p>
              <p>
                <strong>MODEL CONFIDENCE SCORE:</strong>{" "}
                <strong>{result.confidence_percentage || `${displayMainConfidence}%`}</strong>
              </p>
              <hr style={styles.innerDivider} />
              <p><strong>SUMMARY:</strong></p>
              <p>
                The uploaded MRI brain scan has been analyzed with an overall model confidence of{" "}
                <strong>{displayMainConfidence}%</strong>.
              </p>
              {isHemorrhagic ? (
                <p>
                  Features indicative of acute hemorrhage have been detected. The <strong>Grad-CAM</strong>, <strong>Professional Grad-CAM</strong>, and <strong>LIME</strong> explainability maps below visually isolate the high-intensity impact zones.
                </p>
              ) : (
                <p>
                  No acute hemorrhagic regions were identified. Explainability visual maps are hidden for non-hemorrhagic scans.
                </p>
              )}

              <div style={styles.disclaimerBox}>
                <strong>IMPORTANT NOTICE:</strong> This application serves as an educational and research tool. Diagnostic decisions must be reviewed by qualified medical specialists.
              </div>
            </div>
          </section>
        )}

        {/* 4. Model Explainability Gallery */}
        {result && isHemorrhagic && (
          <section style={{ ...styles.card, borderColor: "#fca5a5", backgroundColor: "#fffafa" }}>
            <h2 style={{ ...styles.cardTitleBlue, color: "#b91c1c" }}>
              MODEL EXPLAINABILITY & VISUAL MAPS
            </h2>
            <p style={styles.explainSubtitleText}>
              Exclusively generated for <strong>Hemorrhagic</strong> predictions to isolate target regions.
            </p>

            <div style={styles.explainabilityGrid}>
              {/* Grad-CAM Card */}
              <div style={styles.explainCard}>
                <div style={styles.cardHeaderTag}>HEATMAP</div>
                <h4 style={styles.explainCardTitle}>Grad-CAM Map</h4>
                <div style={styles.imageBox}>
                  <img
                    src={formatImageSource(result.gradcam)}
                    alt="Grad-CAM Map"
                    style={styles.explainImage}
                  />
                  <div style={styles.imageBadge}>CONFIDENCE: {displayMainConfidence}%</div>
                </div>
                <p style={styles.explainDesc}>
                  Highlights regions in the brain scan that directly drove the high-probability classification.
                </p>
              </div>

              {/* Professional Grad-CAM Card */}
              <div style={{ ...styles.explainCard, borderColor: "#ef4444", borderWidth: "2px" }}>
                <div style={{ ...styles.cardHeaderTag, backgroundColor: "#dc2626" }}>REGION FOCUS</div>
                <h4 style={styles.explainCardTitle}>Professional Grad-CAM</h4>
                <div style={styles.imageBox}>
                  <img
                    src={formatImageSource(proGradCamImage)}
                    alt="Pro Grad-CAM"
                    style={styles.explainImage}
                  />
                  <div style={{ ...styles.imageBadge, backgroundColor: "#dc2626" }}>
                    TARGET BOUNDING ZONE
                  </div>
                </div>
                <p style={styles.explainDesc}>
                  Provides an enhanced, high-precision visual focusing tightly on the suspected lesion area.
                </p>
              </div>

              {/* LIME Explanation Card */}
              <div style={styles.explainCard}>
                <div style={{ ...styles.cardHeaderTag, backgroundColor: "#d97706" }}>SUPERPIXELS</div>
                <h4 style={styles.explainCardTitle}>LIME Explanation</h4>
                <div style={styles.imageBox}>
                  <img
                    src={formatImageSource(result.lime)}
                    alt="LIME Explanation"
                    style={styles.explainImage}
                  />
                  <div style={{ ...styles.imageBadge, backgroundColor: "#d97706" }}>
                    FEATURE CONTOURS
                  </div>
                </div>
                <p style={styles.explainDesc}>
                  Identifies key local boundaries and pixel groupings contributing to the model's output.
                </p>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer style={styles.footer}>
        <h3 style={styles.footerTitle}>BRAIN HEMORRHAGE DETECTION SYSTEM</h3>
        <p style={styles.footerSubText}>
          Deep Learning-based MRI Analysis using EfficientNet-B0 with Explainable AI (Grad-CAM & LIME)
        </p>
        <hr style={styles.footerLine} />
        <p style={styles.footerCredits}>
          Developed by <strong>K. Subba Rao</strong>, <strong>D. Kowshik Reddy</strong> & <strong>Sh. Thasleem</strong>
        </p>
        <p style={styles.footerCredits}>Department of Computer Science and Engineering</p>
        <p style={styles.footerCredits}>Lakireddy Bali Reddy College of Engineering</p>
      </footer>
    </div>
  );
}

const styles = {
  appWrapper: {
    backgroundColor: "#0B132B",
    minHeight: "100vh",
    width: "100%",
    boxSizing: "border-box",
    fontFamily: "'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif",
    color: "#F8FAFC",
    margin: 0,
    padding: 0,
  },
  header: {
    background: "linear-gradient(180deg, #1C2541 0%, #0B132B 100%)",
    borderBottom: "1px solid #1E293B",
    padding: "40px 20px 32px 20px",
    textAlign: "center",
  },
  headerTag: {
    display: "inline-block",
    backgroundColor: "#2563EB",
    color: "#FFFFFF",
    fontSize: "11px",
    fontWeight: "900",
    padding: "5px 16px",
    borderRadius: "20px",
    letterSpacing: "1.5px",
    marginBottom: "14px",
    boxShadow: "0 0 12px rgba(37, 99, 235, 0.4)",
  },
  headerTitle: {
    fontSize: "30px",
    fontWeight: "900",
    letterSpacing: "1px",
    margin: "0 0 10px 0",
    color: "#FFFFFF",
  },
  headerSubtitle: {
    fontSize: "14px",
    color: "#94A3B8",
    lineHeight: "1.6",
    margin: 0,
  },
  mainContainer: {
    maxWidth: "1280px",
    margin: "0 auto",
    padding: "32px 24px",
    display: "flex",
    flexDirection: "column",
    gap: "28px",
    boxSizing: "border-box",
  },
  card: {
    backgroundColor: "#FFFFFF",
    color: "#0F172A",
    borderRadius: "16px",
    padding: "32px",
    border: "1px solid #E2E8F0",
    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.3)",
  },
  cardTitleBlue: {
    color: "#1E40AF",
    fontSize: "20px",
    fontWeight: "900",
    letterSpacing: "0.5px",
    margin: "0 0 20px 0",
    textAlign: "center",
  },
  projectHeading: {
    textAlign: "center",
    fontSize: "17px",
    fontWeight: "800",
    color: "#1E293B",
    marginBottom: "24px",
  },
  infoGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    gap: "20px",
  },
  infoBox: {
    backgroundColor: "#F8FAFC",
    padding: "20px",
    borderRadius: "12px",
    border: "1px solid #E2E8F0",
  },
  infoLabel: {
    color: "#64748B",
    fontSize: "11px",
    fontWeight: "900",
    letterSpacing: "1px",
    marginBottom: "6px",
  },
  infoBoldValue: {
    fontSize: "14px",
    fontWeight: "800",
    color: "#0F172A",
    margin: "4px 0",
  },
  infoSubText: {
    fontSize: "12px",
    color: "#64748B",
    margin: 0,
    fontWeight: "600",
  },
  dashboardGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
    gap: "20px",
    width: "100%",
  },
  smallCard: {
    backgroundColor: "#FFFFFF",
    color: "#0F172A",
    borderRadius: "16px",
    padding: "24px",
    border: "1px solid #E2E8F0",
    boxShadow: "0 10px 20px -5px rgba(0, 0, 0, 0.2)",
    display: "flex",
    flexDirection: "column",
    justify: "space-between",
  },
  uploadForm: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "14px",
  },
  dashedUploadBox: {
    border: "2px dashed #94A3B8",
    backgroundColor: "#F1F5F9",
    padding: "16px",
    borderRadius: "12px",
    width: "100%",
    textAlign: "center",
    boxSizing: "border-box",
  },
  fileNameBadge: {
    backgroundColor: "#EFF6FF",
    color: "#1E40AF",
    padding: "6px 12px",
    borderRadius: "6px",
    fontSize: "12px",
    border: "1px solid #BFDBFE",
    textAlign: "center",
    wordBreak: "break-all",
  },
  previewContainer: {
    border: "2px solid #CBD5E1",
    borderRadius: "12px",
    padding: "4px",
    backgroundColor: "#F8FAFC",
  },
  previewImage: {
    width: "150px",
    height: "150px",
    objectFit: "cover",
    borderRadius: "8px",
    display: "block",
  },
  predictBtn: {
    backgroundColor: "#1E40AF",
    color: "#FFFFFF",
    border: "none",
    padding: "12px 20px",
    borderRadius: "8px",
    fontWeight: "800",
    fontSize: "14px",
    letterSpacing: "0.5px",
    flex: 1,
    boxShadow: "0 4px 12px rgba(30, 64, 175, 0.3)",
    transition: "all 0.2s ease-in-out",
  },
  resetBtn: {
    backgroundColor: "#475569",
    color: "#FFFFFF",
    border: "none",
    padding: "12px 16px",
    borderRadius: "8px",
    fontWeight: "800",
    fontSize: "13px",
    cursor: "pointer",
    transition: "all 0.2s ease-in-out",
  },
  resultContainer: {
    display: "flex",
    flexDirection: "column",
    justify: "center",
    flex: 1,
  },
  resultRow: {
    display: "flex",
    justify: "space-between",
    alignItems: "center",
    padding: "14px 0",
    borderBottom: "1px solid #F1F5F9",
  },
  rowLabel: {
    fontSize: "12px",
    fontWeight: "900",
    color: "#64748B",
    letterSpacing: "0.5px",
  },
  statusTag: {
    padding: "6px 14px",
    borderRadius: "8px",
    fontSize: "13px",
    fontWeight: "900",
    border: "1px solid",
    letterSpacing: "0.5px",
  },
  confidenceText: {
    fontSize: "20px",
    fontWeight: "900",
    color: "#1D4ED8",
  },
  probContainer: {
    display: "flex",
    flexDirection: "column",
    justify: "center",
    flex: 1,
  },
  probHeader: {
    display: "flex",
    justify: "space-between",
    fontSize: "13px",
    fontWeight: "800",
  },
  progressTrack: {
    backgroundColor: "#E2E8F0",
    borderRadius: "8px",
    height: "12px",
    width: "100%",
    overflow: "hidden",
    marginTop: "6px",
  },
  progressFill: {
    height: "100%",
    transition: "width 0.4s ease-in-out",
  },
  emptyStateBox: {
    display: "flex",
    alignItems: "center",
    justify: "center",
    flex: 1,
    minHeight: "120px",
  },
  interpretationText: {
    fontSize: "14px",
    lineHeight: "1.7",
    color: "#334155",
  },
  innerDivider: {
    border: "none",
    borderTop: "1px solid #E2E8F0",
    margin: "16px 0",
  },
  disclaimerBox: {
    marginTop: "20px",
    padding: "14px",
    backgroundColor: "#F0F9FF",
    borderLeft: "5px solid #0284C7",
    borderRadius: "6px",
    fontSize: "12px",
    color: "#0369A1",
    fontWeight: "600",
  },
  explainSubtitleText: {
    textAlign: "center",
    color: "#64748B",
    fontSize: "13px",
    marginTop: "-12px",
    marginBottom: "28px",
  },
  explainabilityGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    gap: "24px",
  },
  explainCard: {
    backgroundColor: "#FFFFFF",
    border: "1px solid #CBD5E1",
    borderRadius: "14px",
    padding: "20px",
    position: "relative",
    boxShadow: "0 10px 15px -3px rgba(0,0,0,0.05)",
  },
  cardHeaderTag: {
    position: "absolute",
    top: "16px",
    right: "16px",
    backgroundColor: "#1E40AF",
    color: "#FFFFFF",
    fontSize: "10px",
    fontWeight: "900",
    padding: "4px 8px",
    borderRadius: "6px",
    letterSpacing: "0.5px",
  },
  explainCardTitle: {
    fontSize: "16px",
    fontWeight: "900",
    color: "#0F172A",
    margin: "0 0 16px 0",
  },
  imageBox: {
    position: "relative",
    marginBottom: "14px",
  },
  explainImage: {
    width: "100%",
    height: "220px",
    objectFit: "cover",
    borderRadius: "10px",
    backgroundColor: "#000000",
    display: "block",
  },
  imageBadge: {
    position: "absolute",
    bottom: "8px",
    left: "8px",
    backgroundColor: "rgba(15, 23, 42, 0.85)",
    color: "#FFFFFF",
    fontSize: "10px",
    fontWeight: "900",
    padding: "4px 8px",
    borderRadius: "4px",
    letterSpacing: "0.5px",
  },
  explainDesc: {
    fontSize: "12px",
    color: "#475569",
    lineHeight: "1.5",
    margin: 0,
    fontWeight: "600",
  },
  errorCard: {
    padding: "16px",
    backgroundColor: "#FEF2F2",
    border: "1px solid #FCA5A5",
    color: "#991B1B",
    borderRadius: "12px",
    fontSize: "14px",
  },
  footer: {
    backgroundColor: "#030712",
    borderTop: "1px solid #1E293B",
    padding: "40px 20px",
    textAlign: "center",
    marginTop: "20px",
  },
  footerTitle: {
    fontSize: "18px",
    fontWeight: "900",
    color: "#FFFFFF",
    margin: "0 0 6px 0",
    letterSpacing: "1px",
  },
  footerSubText: {
    fontSize: "12px",
    color: "#94A3B8",
    margin: "0 0 20px 0",
  },
  footerLine: {
    width: "80px",
    border: "none",
    borderTop: "1px solid #334155",
    margin: "0 auto 20px auto",
  },
  footerCredits: {
    fontSize: "12px",
    color: "#CBD5E1",
    margin: "4px 0",
  },
};

export default App;
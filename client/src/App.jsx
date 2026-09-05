import React, { useState } from "react";
import { predictImage } from "./services/api";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "https://brain-hemorrhage-backend.onrender.com";

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
        err.response?.data?.detail ||
          err.message ||
          "Failed to process scan on server."
      );
      console.error("Prediction Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const isHemorrhagic = result?.prediction === "Hemorrhagic";

  const parsePercent = (val) => {
    if (val === undefined || val === null) return 0;
    if (typeof val === "number") return val;
    return parseFloat(String(val).replace("%", "")) || 0;
  };

  const resolveImageUrl = (pathOrData) => {
    if (!pathOrData) return "";
    if (pathOrData.startsWith("data:") || pathOrData.startsWith("http")) {
      return pathOrData;
    }
    const cleanPath = pathOrData.startsWith("/") ? pathOrData : `/${pathOrData}`;
    return `${API_BASE_URL}${cleanPath}`;
  };

  const mainConfidence = parsePercent(result?.confidence);
  const hemorrhagicPct = parsePercent(result?.hemorrhage_confidence);
  const nonHemorrhagicPct = parsePercent(result?.normal_confidence);

  return (
    <div style={styles.appWrapper}>
      <header style={styles.header}>
        <h1 style={styles.headerTitle}>Brain Hemorrhage Detection System</h1>
        <p style={styles.headerSubtitle}>
          Deep learning-based MRI analysis using EfficientNet-B0 &middot; FastAPI
          &middot; React &middot; Grad-CAM &middot; LIME explainability
        </p>
      </header>

      <main style={styles.mainContainer}>
        <section style={styles.card}>
          <h2 style={styles.cardTitle}>Project Information</h2>
          <h3 style={styles.projectHeading}>
            A Deep Neural Network Approach for Hemorrhagic Stroke Identification
            in MRI Scans
          </h3>
          <div style={styles.infoGrid}>
            <div style={styles.infoBox}>
              <p style={styles.infoLabel}>Under the guidance of</p>
              <p style={styles.infoBoldValue}>Mr. N. Srinivasa Rao</p>
              <p style={styles.infoSubText}>Sr. Assistant Professor</p>

              <div style={{ marginTop: "16px" }}>
                <p style={styles.infoLabel}>Department</p>
                <p style={styles.infoBoldValue}>
                  Computer Science and Engineering
                </p>
              </div>
            </div>

            <div style={styles.infoBox}>
              <p style={styles.infoLabel}>Presented by</p>
              <p style={styles.infoBoldValue}>
                K. Subba Rao &nbsp;&middot;&nbsp; 23761A0599
              </p>
              <p style={styles.infoBoldValue}>
                D. Kowshik Reddy &nbsp;&middot;&nbsp; 23761A0584
              </p>
              <p style={styles.infoBoldValue}>
                Sh. Thasleem &nbsp;&middot;&nbsp; 23761A05C2
              </p>

              <div style={{ marginTop: "16px" }}>
                <p style={styles.infoLabel}>College</p>
                <p style={styles.infoBoldValue}>
                  Lakireddy Bali Reddy College of Engineering
                </p>
              </div>
            </div>
          </div>
        </section>

        <div style={styles.dashboardGrid}>
          <div style={styles.smallCard}>
            <h3 style={styles.cardTitle}>Upload MRI Brain Scan</h3>
            <form onSubmit={handleSubmit} style={styles.uploadForm}>
              <div style={styles.dashedUploadBox}>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  style={{ width: "100%", cursor: "pointer", fontSize: "13px" }}
                />
              </div>

              {fileName && <p style={styles.fileNameText}>{fileName}</p>}

              {preview && (
                <img
                  src={preview}
                  alt="MRI preview"
                  style={styles.previewImage}
                />
              )}

              <div style={{ width: "100%", display: "flex", gap: "10px" }}>
                <button
                  type="submit"
                  disabled={loading || !selectedFile}
                  style={{
                    ...styles.predictBtn,
                    opacity: loading || !selectedFile ? 0.5 : 1,
                    cursor: loading || !selectedFile ? "not-allowed" : "pointer",
                  }}
                >
                  {loading ? "Analyzing MRI..." : "Predict"}
                </button>

                {(selectedFile || result) && (
                  <button
                    type="button"
                    onClick={handleReset}
                    style={styles.resetBtn}
                  >
                    Reset
                  </button>
                )}
              </div>
            </form>
          </div>

          <div style={styles.smallCard}>
            <h3 style={styles.cardTitle}>Prediction Result</h3>
            {result ? (
              <div style={styles.resultContainer}>
                <div style={styles.resultRow}>
                  <span style={styles.rowLabel}>Prediction</span>
                  <span
                    style={{
                      ...styles.value,
                      color: isHemorrhagic ? "#dc2626" : "#16a34a",
                    }}
                  >
                    {result.prediction}
                  </span>
                </div>
                <div style={styles.resultRow}>
                  <span style={styles.rowLabel}>Confidence</span>
                  <strong style={styles.confidenceText}>
                    {mainConfidence.toFixed(2)}%
                  </strong>
                </div>
              </div>
            ) : (
              <div style={styles.emptyStateBox}>
                <p style={styles.emptyStateText}>Awaiting scan upload...</p>
              </div>
            )}
          </div>

          <div style={styles.smallCard}>
            <h3 style={styles.cardTitle}>Class Probabilities</h3>
            {result ? (
              <div style={styles.probContainer}>
                <div style={styles.probHeader}>
                  <span>Hemorrhagic</span>
                  <span>{hemorrhagicPct.toFixed(2)}%</span>
                </div>
                <div style={styles.progressTrack}>
                  <div
                    style={{
                      ...styles.progressFill,
                      width: `${hemorrhagicPct}%`,
                      backgroundColor: "#ef4444",
                    }}
                  />
                </div>

                <div style={{ ...styles.probHeader, marginTop: "20px" }}>
                  <span>NonHemorrhagic</span>
                  <span>{nonHemorrhagicPct.toFixed(2)}%</span>
                </div>
                <div style={styles.progressTrack}>
                  <div
                    style={{
                      ...styles.progressFill,
                      width: `${nonHemorrhagicPct}%`,
                      backgroundColor: "#2563eb",
                    }}
                  />
                </div>
              </div>
            ) : (
              <div style={styles.emptyStateBox}>
                <p style={styles.emptyStateText}>
                  Probabilities will appear here.
                </p>
              </div>
            )}
          </div>
        </div>

        {error && (
          <div style={styles.errorCard}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <section style={styles.card}>
            <h2 style={styles.cardTitle}>Medical Interpretation</h2>
            <div style={styles.interpretationText}>
              <p>
                <strong>Prediction:</strong> {result.prediction}
              </p>
              <p>
                <strong>Confidence:</strong> {mainConfidence.toFixed(2)}%
              </p>

              {isHemorrhagic ? (
                <>
                  <p>
                    The uploaded MRI scan has been classified as{" "}
                    <strong>Hemorrhagic</strong> with a confidence score of{" "}
                    <strong>{mainConfidence.toFixed(2)}%</strong>.
                  </p>
                  <p>
                    The <strong>Grad-CAM</strong> visualization highlights the
                    image regions that most influenced the model's prediction.
                  </p>
                  <p>
                    The <strong>Professional Grad-CAM</strong> provides an
                    enhanced visualization of the region that contributed most
                    strongly to the classification.
                  </p>
                  <p>
                    The <strong>LIME explanation</strong> identifies the local
                    image features that had the greatest influence on the
                    prediction.
                  </p>
                </>
              ) : (
                <>
                  <p>
                    The uploaded MRI scan shows no visual indicators consistent
                    with hemorrhage, and has been classified as{" "}
                    <strong>NonHemorrhagic</strong> with a confidence score of{" "}
                    <strong>{mainConfidence.toFixed(2)}%</strong>.
                  </p>
                  <p>
                    <strong>Estimated hemorrhage likelihood:</strong>{" "}
                    {hemorrhagicPct.toFixed(2)}%
                  </p>
                  <p>
                    This is a low estimated likelihood based on the model's
                    analysis of this single scan, not a clinical risk prediction
                    for future events.
                  </p>
                </>
              )}

              <div style={styles.disclaimerBox}>
                <strong>Note:</strong> This application is intended for educational
                and research purposes. The predictions should support, not replace,
                clinical judgment by qualified healthcare professionals.
              </div>
            </div>
          </section>
        )}

        {result && isHemorrhagic && (
          <section style={styles.card}>
            <h2 style={styles.cardTitle}>Model Explainability</h2>

            <div style={styles.explainabilityGrid}>
              <div style={styles.explainCard}>
                <h4 style={styles.explainCardTitle}>Grad-CAM</h4>
                <img
                  src={resolveImageUrl(result.gradcam)}
                  alt="Grad-CAM"
                  style={styles.explainImage}
                />
                <p style={styles.explainDesc}>
                  Highlights the MRI regions that contributed most to the model's
                  prediction.
                </p>
              </div>

              <div style={styles.explainCard}>
                <h4 style={styles.explainCardTitle}>Professional Grad-CAM</h4>
                <img
                  src={resolveImageUrl(result.pro_gradcam)}
                  alt="Professional Grad-CAM"
                  style={styles.explainImage}
                />
                <p style={styles.explainDesc}>
                  Provides a refined visualization of the important regions used by
                  the deep learning model.
                </p>
              </div>

              <div style={styles.explainCard}>
                <h4 style={styles.explainCardTitle}>LIME Explanation</h4>
                <img
                  src={resolveImageUrl(result.lime)}
                  alt="LIME Explanation"
                  style={styles.explainImage}
                />
                <p style={styles.explainDesc}>
                  Explains the local image features that influenced the final
                  classification decision.
                </p>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer style={styles.footer}>
        <h3 style={styles.footerTitle}>Brain Hemorrhage Detection System</h3>
        <p style={styles.footerSubText}>
          Deep learning-based MRI analysis using EfficientNet-B0 with explainable
          AI (Grad-CAM & LIME)
        </p>
        <hr style={styles.footerLine} />
        <p style={styles.footerCredits}>
          Developed by K. Subba Rao, D. Kowshik Reddy & Sh. Thasleem
        </p>
        <p style={styles.footerCredits}>
          Department of Computer Science and Engineering
        </p>
        <p style={styles.footerCredits}>
          Lakireddy Bali Reddy College of Engineering
        </p>
      </footer>
    </div>
  );
}

const styles = {
  appWrapper: {
    backgroundColor: "#f4f7fb",
    minHeight: "100vh",
    width: "100%",
    boxSizing: "border-box",
    fontFamily:
      "'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif",
    color: "#0f172a",
    margin: 0,
    padding: 0,
  },
  header: {
    background: "#1e40af",
    padding: "36px 20px",
    textAlign: "center",
  },
  headerTitle: {
    fontSize: "28px",
    fontWeight: "700",
    margin: "0 0 10px 0",
    color: "#ffffff",
  },
  headerSubtitle: {
    fontSize: "14px",
    color: "#dbeafe",
    lineHeight: "1.6",
    margin: 0,
  },
  mainContainer: {
    maxWidth: "1200px",
    margin: "0 auto",
    padding: "32px 24px",
    display: "flex",
    flexDirection: "column",
    gap: "24px",
    boxSizing: "border-box",
  },
  card: {
    backgroundColor: "#ffffff",
    borderRadius: "12px",
    padding: "28px",
    border: "1px solid #dbeafe",
  },
  cardTitle: {
    color: "#1e40af",
    fontSize: "20px",
    fontWeight: "700",
    margin: "0 0 20px 0",
    textAlign: "center",
  },
  projectHeading: {
    textAlign: "center",
    fontSize: "16px",
    fontWeight: "600",
    color: "#1e293b",
    marginBottom: "20px",
  },
  infoGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    gap: "16px",
  },
  infoBox: {
    backgroundColor: "#f8fafc",
    padding: "16px",
    borderRadius: "10px",
    border: "1px solid #e2e8f0",
  },
  infoLabel: {
    color: "#64748b",
    fontSize: "12px",
    fontWeight: "600",
    marginBottom: "4px",
  },
  infoBoldValue: {
    fontSize: "14px",
    fontWeight: "600",
    color: "#0f172a",
    margin: "2px 0",
  },
  infoSubText: {
    fontSize: "12px",
    color: "#64748b",
    margin: 0,
  },
  dashboardGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
    gap: "20px",
    width: "100%",
  },
  smallCard: {
    backgroundColor: "#ffffff",
    borderRadius: "12px",
    padding: "24px",
    border: "1px solid #dbeafe",
    display: "flex",
    flexDirection: "column",
  },
  uploadForm: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "14px",
  },
  dashedUploadBox: {
    border: "2px dashed #93c5fd",
    backgroundColor: "#f8fafc",
    padding: "14px",
    borderRadius: "10px",
    width: "100%",
    textAlign: "center",
    boxSizing: "border-box",
  },
  fileNameText: {
    fontSize: "13px",
    fontWeight: "600",
    color: "#1e40af",
    margin: 0,
    wordBreak: "break-all",
    textAlign: "center",
  },
  previewImage: {
    width: "100%",
    maxHeight: "220px",
    objectFit: "contain",
    borderRadius: "10px",
    border: "1px solid #dbeafe",
    backgroundColor: "#ffffff",
  },
  predictBtn: {
    backgroundColor: "#1e40af",
    color: "#ffffff",
    border: "none",
    padding: "12px 20px",
    borderRadius: "8px",
    fontWeight: "600",
    fontSize: "14px",
    flex: 1,
  },
  resetBtn: {
    backgroundColor: "#64748b",
    color: "#ffffff",
    border: "none",
    padding: "12px 16px",
    borderRadius: "8px",
    fontWeight: "600",
    fontSize: "13px",
    cursor: "pointer",
  },
  resultContainer: {
    display: "flex",
    flexDirection: "column",
    flex: 1,
  },
  resultRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "12px 0",
    borderBottom: "1px solid #f1f5f9",
  },
  rowLabel: {
    fontSize: "13px",
    color: "#64748b",
    fontWeight: "600",
  },
  value: {
    fontSize: "15px",
    fontWeight: "700",
  },
  confidenceText: {
    fontSize: "18px",
    fontWeight: "700",
    color: "#1e40af",
  },
  probContainer: {
    display: "flex",
    flexDirection: "column",
    flex: 1,
  },
  probHeader: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: "13px",
    fontWeight: "600",
  },
  progressTrack: {
    backgroundColor: "#e2e8f0",
    borderRadius: "8px",
    height: "10px",
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
    justifyContent: "center",
    flex: 1,
    minHeight: "100px",
  },
  emptyStateText: {
    margin: 0,
    color: "#94a3b8",
    fontSize: "14px",
    fontWeight: "600",
  },
  interpretationText: {
    fontSize: "14px",
    lineHeight: "1.7",
    color: "#334155",
  },
  disclaimerBox: {
    marginTop: "18px",
    padding: "14px",
    backgroundColor: "#eff6ff",
    borderLeft: "4px solid #1e40af",
    borderRadius: "6px",
    fontSize: "12px",
    color: "#1e3a8a",
  },
  explainabilityGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
    gap: "20px",
  },
  explainCard: {
    backgroundColor: "#f8fafc",
    border: "1px solid #dbeafe",
    borderRadius: "12px",
    padding: "16px",
  },
  explainCardTitle: {
    fontSize: "15px",
    fontWeight: "700",
    color: "#1e40af",
    margin: "0 0 12px 0",
    textAlign: "center",
  },
  explainImage: {
    width: "100%",
    borderRadius: "8px",
    display: "block",
    marginBottom: "10px",
  },
  explainDesc: {
    fontSize: "12px",
    color: "#475569",
    lineHeight: "1.5",
    margin: 0,
  },
  errorCard: {
    padding: "14px",
    backgroundColor: "#fef2f2",
    border: "1px solid #fca5a5",
    color: "#991b1b",
    borderRadius: "10px",
    fontSize: "14px",
  },
  footer: {
    backgroundColor: "#1e293b",
    padding: "32px 20px",
    textAlign: "center",
    marginTop: "12px",
  },
  footerTitle: {
    fontSize: "16px",
    fontWeight: "700",
    color: "#ffffff",
    margin: "0 0 6px 0",
  },
  footerSubText: {
    fontSize: "12px",
    color: "#94a3b8",
    margin: "0 0 16px 0",
  },
  footerLine: {
    width: "60px",
    border: "none",
    borderTop: "1px solid #334155",
    margin: "0 auto 16px auto",
  },
  footerCredits: {
    fontSize: "12px",
    color: "#cbd5e1",
    margin: "3px 0",
  },
};

export default App;
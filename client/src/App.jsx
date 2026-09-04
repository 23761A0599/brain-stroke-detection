import React, { useState } from "react";

const API_URL = "https://brain-hemorrhage-backend.onrender.com/predict";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError("");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Server error ${response.status}`);
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to process image on server.");
      console.error("Analysis Error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h2 style={styles.title}>Brain Hemorrhage Detection</h2>
        <p style={styles.subtitle}>Upload a CT / MRI scan to analyze for hemorrhage</p>

        <form onSubmit={handleSubmit} style={styles.form}>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={styles.fileInput}
          />

          {preview && (
            <div style={styles.previewContainer}>
              <img src={preview} alt="MRI Preview" style={styles.previewImage} />
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !selectedFile}
            style={{
              ...styles.button,
              opacity: loading || !selectedFile ? 0.6 : 1,
              cursor: loading || !selectedFile ? "not-allowed" : "pointer"
            }}
          >
            {loading ? "Analyzing Image..." : "Upload & Analyze"}
          </button>
        </form>

        {error && <div style={styles.errorBox}>{error}</div>}

        {result && (
          <div style={{
            ...styles.resultBox,
            borderColor: result.prediction === "Hemorrhagic" ? "#ef4444" : "#10b981",
            backgroundColor: result.prediction === "Hemorrhagic" ? "#fef2f2" : "#ecfdf5"
          }}>
            <h3 style={styles.resultTitle}>Prediction Result</h3>
            <p style={styles.resultText}>
              <strong>Diagnosis:</strong>{" "}
              <span style={{
                color: result.prediction === "Hemorrhagic" ? "#dc2626" : "#059669",
                fontWeight: "bold"
              }}>
                {result.prediction}
              </span>
            </p>
            <p style={styles.resultText}>
              <strong>Confidence:</strong> {result.confidence}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    backgroundColor: "#f3f4f6",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "20px",
    fontFamily: "'Segoe UI', Roboto, sans-serif"
  },
  card: {
    backgroundColor: "#ffffff",
    padding: "32px",
    borderRadius: "12px",
    boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
    maxWidth: "480px",
    width: "100%",
    textAlign: "center"
  },
  title: {
    fontSize: "24px",
    fontWeight: "700",
    color: "#1f2937",
    margin: "0 0 8px 0"
  },
  subtitle: {
    fontSize: "14px",
    color: "#6b7280",
    margin: "0 0 24px 0"
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "16px"
  },
  fileInput: {
    padding: "10px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    backgroundColor: "#f9fafb",
    fontSize: "14px"
  },
  previewContainer: {
    display: "flex",
    justifyContent: "center",
    margin: "8px 0"
  },
  previewImage: {
    width: "224px",
    height: "224px",
    objectFit: "cover",
    borderRadius: "8px",
    border: "1px solid #e5e7eb"
  },
  button: {
    backgroundColor: "#2563eb",
    color: "#ffffff",
    border: "none",
    padding: "12px",
    borderRadius: "6px",
    fontSize: "16px",
    fontWeight: "600"
  },
  errorBox: {
    marginTop: "16px",
    padding: "12px",
    borderRadius: "6px",
    backgroundColor: "#fef2f2",
    color: "#dc2626",
    fontSize: "14px",
    border: "1px solid #fca5a5"
  },
  resultBox: {
    marginTop: "20px",
    padding: "16px",
    borderRadius: "8px",
    borderWidth: "1px",
    borderStyle: "solid",
    textAlign: "left"
  },
  resultTitle: {
    margin: "0 0 8px 0",
    fontSize: "18px",
    color: "#111827"
  },
  resultText: {
    margin: "4px 0",
    fontSize: "15px",
    color: "#374151"
  }
};

export default App;
import { useState } from "react";
import LoadingSpinner from "./LoadingSpinner";
import MedicalInterpretation from "./MedicalInterpretation";
import { predictImage } from "../services/api";
import PredictionCard from "./PredictionCard";
import ProbabilityChart from "./ProbabilityChart";
import ExplanationCard from "./ExplanationCard";
import "../styles/UploadCard.css";
import "../styles/Dashboard.css";

function UploadCard() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      alert("Please select an MRI image.");
      return;
    }

    try {
      setLoading(true);
      const response = await predictImage(selectedFile);
      setResult(response);
    } catch (error) {
      console.error(error);
      alert("Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Top Dashboard Row */}
      <div className="dashboard-grid">

        {/* Upload Card */}
        <div className="upload-card">
          <h2>Upload MRI Brain Scan</h2>

          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />

          {selectedFile && (
            <>
              <p
                style={{
                  marginTop: "15px",
                  fontWeight: "600",
                  color: "#0f4c81",
                }}
              >
                {selectedFile.name}
              </p>

              <img
                src={preview}
                alt="MRI Preview"
                style={{
                  width: "100%",
                  marginTop: "20px",
                  borderRadius: "12px",
                  border: "2px solid #dbeafe",
                  maxHeight: "320px",
                  objectFit: "contain",
                  background: "#ffffff",
                }}
              />
            </>
          )}

          <button
            onClick={handlePredict}
            disabled={loading}
            style={{ marginTop: "25px" }}
          >
            {loading ? "Analyzing MRI..." : "Predict"}
          </button>
        </div>

        {/* Prediction */}
        {result && !loading ? (
          <>
            <PredictionCard result={result} />

            <ProbabilityChart
              probabilities={result.class_probabilities || result.probabilities}
            />
          </>
        ) : (
          <>
            <div></div>
            <div></div>
          </>
        )}

      </div>

      {/* Loading */}
      {loading && <LoadingSpinner />}

      {/* Medical Interpretation */}
      {result && !loading && (
        <div className="full-width">
          <MedicalInterpretation result={result} />
        </div>
      )}

      {/* Explainability Section - Rendered for all predictions */}
      {result && !loading && (
        <div className="full-width">
          <ExplanationCard result={result} />
        </div>
      )}
    </>
  );
}

export default UploadCard;
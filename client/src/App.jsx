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

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError("Failed to connect to the backend server. It may be waking up from cold start.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "600px", margin: "40px auto", padding: "20px", fontFamily: "sans-serif" }}>
      <h2>Brain Hemorrhage Detection</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <br /><br />
        {preview && (
          <div>
            <img src={preview} alt="MRI Preview" style={{ width: "224px", height: "224px", objectFit: "cover" }} />
          </div>
        )}
        <br />
        <button type="submit" disabled={loading || !selectedFile} style={{ padding: "10px 20px" }}>
          {loading ? "Analyzing Image..." : "Upload & Analyze"}
        </button>
      </form>

      {error && <p style={{ color: "red", marginTop: "20px" }}>{error}</p>}

      {result && (
        <div style={{ marginTop: "20px", padding: "15px", border: "1px solid #ccc", borderRadius: "5px" }}>
          <h3>Prediction Result</h3>
          <p><strong>Diagnosis:</strong> {result.prediction}</p>
          <p><strong>Confidence:</strong> {result.confidence}%</p>
        </div>
      )}
    </div>
  );
}

export default App;
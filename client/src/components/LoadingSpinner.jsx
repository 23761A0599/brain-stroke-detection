import "../styles/LoadingSpinner.css";

function LoadingSpinner() {
  return (
    <div className="loading-container">

      <div className="spinner"></div>

      <h3>Analyzing MRI Scan...</h3>

      <p>Please wait while the AI model processes the image.</p>

    </div>
  );
}

export default LoadingSpinner;
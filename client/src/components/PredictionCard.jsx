import "../styles/PredictionCard.css";

function PredictionCard({ result }) {

  const predictionClass =
    result.prediction === "Hemorrhagic"
      ? "hemorrhagic"
      : "nonhemorrhagic";

  return (
    <div className="prediction-card">

      <h2>Prediction Result</h2>

      <div className="prediction-row">
        <span className="label">Prediction</span>

        <span className={`value ${predictionClass}`}>
          {result.prediction}
        </span>
      </div>

      <div className="prediction-row">
        <span className="label">Confidence</span>

        <span className="value">
          {Number(result.confidence).toFixed(2)}%
        </span>
      </div>

    </div>
  );
}

export default PredictionCard;
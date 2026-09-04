import "../styles/ProbabilityChart.css";

function ProbabilityChart({ probabilities }) {
  if (!probabilities) return null;

  return (
    <div className="probability-card">
      <h2>Class Probabilities</h2>

      {Object.entries(probabilities).map(([label, value]) => (
        <div key={label} className="probability-item">
          <div className="probability-header">
            <span>{label}</span>
            <span>{Number(value).toFixed(2)}%</span>
          </div>

          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${value}%` }}
            ></div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default ProbabilityChart;
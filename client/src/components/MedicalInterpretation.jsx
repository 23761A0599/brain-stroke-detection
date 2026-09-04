import "../styles/MedicalInterpretation.css";

function MedicalInterpretation({ result }) {
  if (!result) return null;

  const prediction = result.prediction;
  const confidence = Number(result.confidence).toFixed(2);

  return (
    <div className="medical-card">

      <h2>Medical Interpretation</h2>

      <div className="medical-content">

        <p>
          <strong>Prediction:</strong> {prediction}
        </p>

        <p>
          <strong>Confidence:</strong> {confidence}%
        </p>

        <p>
          <strong>Model Interpretation:</strong>
        </p>

        <p>
          The uploaded MRI scan has been classified as
          <strong> {prediction}</strong> with a confidence score of
          <strong> {confidence}%</strong>.
        </p>

        <p>
          The <strong>Grad-CAM</strong> visualization highlights the image
          regions that most influenced the model's prediction.
        </p>

        <p>
          The <strong>Professional Grad-CAM</strong> provides an enhanced
          visualization of the region that contributed most strongly to the
          classification.
        </p>

        <p>
          The <strong>LIME explanation</strong> identifies the local image
          features that had the greatest influence on the prediction.
        </p>

        <div className="medical-note">
          <strong>Note:</strong> This application is intended for educational
          and research purposes. The predictions should support, not replace,
          clinical judgment by qualified healthcare professionals.
        </div>

      </div>

    </div>
  );
}

export default MedicalInterpretation;
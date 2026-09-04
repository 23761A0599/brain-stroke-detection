import "../styles/ExplanationCard.css";

function ExplanationCard({ result }) {
  // Helper function to extract or construct clean image src
  const getImageSrc = (path) => {
    if (!path) return "";
    if (path.startsWith("http")) return path;
    const cleanPath = path.replace("\\", "/").replace(/^\/+/, "");
    return `http://localhost:8000/${cleanPath}`;
  };

  const gradcamSrc = getImageSrc(
    result?.gradcam_url || result?.gradcam
  );
  const gradcamProSrc = getImageSrc(
    result?.gradcam_pro_url || result?.professional_gradcam || result?.gradcam_url
  );
  const limeSrc = getImageSrc(
    result?.lime_url || result?.lime
  );

  return (
    <div className="explanation-container">

      <h2>Model Explainability</h2>

      <div className="image-grid">

        <div className="image-card">

          <h3>Grad-CAM</h3>

          <img
            src={gradcamSrc}
            alt="GradCAM"
          />

          <p className="image-description">
            Highlights the MRI regions that contributed most to the model's
            prediction.
          </p>

        </div>

        <div className="image-card">

          <h3>Professional Grad-CAM</h3>

          <img
            src={gradcamProSrc}
            alt="Professional GradCAM"
          />

          <p className="image-description">
            Provides a refined visualization of the important regions used by
            the deep learning model.
          </p>

        </div>

        <div className="image-card">

          <h3>LIME Explanation</h3>

          <img
            src={limeSrc}
            alt="LIME"
          />

          <p className="image-description">
            Explains the local image features that influenced the final
            classification decision.
          </p>

        </div>

      </div>

    </div>
  );
}

export default ExplanationCard;
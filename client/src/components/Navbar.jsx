import "../styles/Navbar.css";

function Navbar() {
  return (
    <header className="navbar">

      <h1>🧠 Brain Hemorrhage Detection System</h1>

      <p className="navbar-subtitle">
        Deep Learning-based MRI Analysis using EfficientNet-B0
      </p>

      <p className="navbar-tech">
        FastAPI • React • Grad-CAM • LIME Explainability
      </p>

    </header>
  );
}

export default Navbar;
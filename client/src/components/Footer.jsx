import "../styles/Footer.css";

function Footer() {
  return (
    <footer className="footer">

      <div className="footer-content">

        <h3>Brain Hemorrhage Detection System</h3>

        <p>
          Deep Learning-based MRI Analysis using EfficientNet-B0 with
          Explainable AI (Grad-CAM & LIME)
        </p>

        <div className="footer-line"></div>

        <p>
          Developed by
          <strong> K. Subba Rao, D. Kowshik Reddy & Sh. Thasleem</strong>
        </p>

        <p>
          Department of Computer Science and Engineering
        </p>

        <p>
          Lakireddy Bali Reddy College of Engineering
        </p>

        <small>
          © 2026 Brain Hemorrhage Detection System | Academic Project
        </small>

      </div>

    </footer>
  );
}

export default Footer;
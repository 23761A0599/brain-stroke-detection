import "../styles/ProjectInfo.css";

function ProjectInfo() {
  return (
    <section className="project-info">

      <h2>Project Information</h2>

      <div className="project-title">

        <h1>
          A Deep Neural Network Approach for Hemorrhagic
          Stroke Identification in MRI Scans
        </h1>

      </div>

      <div className="project-grid">

        <div className="project-card">

          <h3>👨‍🏫 Under the Guidance of</h3>

          <p><strong>Mr. N. Srinivasa Rao</strong></p>

          <p>Sr. Assistant Professor</p>

        </div>

        <div className="project-card">

          <h3>👨‍💻 Presented By</h3>

          <table>

            <tbody>

              <tr>
                <td>K. Subba Rao</td>
                <td>23761A0599</td>
              </tr>

              <tr>
                <td>D. Kowshik Reddy</td>
                <td>23761A0584</td>
              </tr>

              <tr>
                <td>Sh. Thasleem</td>
                <td>23761A05C2</td>
              </tr>

            </tbody>

          </table>

        </div>

        <div className="project-card">

          <h3>🏫 Department</h3>

          <p>Computer Science and Engineering</p>

        </div>

        <div className="project-card">

          <h3>🎓 College</h3>

          <p>Lakireddy Bali Reddy College of Engineering</p>

        </div>

      </div>

    </section>
  );
}

export default ProjectInfo;
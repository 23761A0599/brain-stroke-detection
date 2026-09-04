import Navbar from "../components/Navbar";
import ProjectInfo from "../components/ProjectInfo";
import UploadCard from "../components/UploadCard";
import Footer from "../components/Footer";

import "../styles/Home.css";

function Home() {
  return (
    <div className="home">

      <Navbar />

      <ProjectInfo />

      <main className="main-content">

        <UploadCard />

      </main>

      <Footer />

    </div>
  );
}

export default Home;
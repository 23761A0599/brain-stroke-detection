import { useState } from "react";

const ALLOWED_USERS = [
  "nsr543@gmail.com",
  "yadavsubbu048@gmail.com",
  "kowshikreddydesam@gmail.com",
];
const SITE_PASSWORD = "2023-2027";

export default function PasswordGate({ children }) {
  const [unlocked, setUnlocked] = useState(
    sessionStorage.getItem("bhd_unlocked") === "true"
  );
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    const cleanUser = username.trim().toLowerCase();
    const validUser = ALLOWED_USERS.includes(cleanUser);
    const validPass = password === SITE_PASSWORD;

    if (validUser && validPass) {
      sessionStorage.setItem("bhd_unlocked", "true");
      setUnlocked(true);
      setError(false);
    } else {
      setError(true);
    }
  };

  if (unlocked) return children;

  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", height: "100vh", fontFamily: "sans-serif",
      background: "#f5f7fb"
    }}>
      <form onSubmit={handleSubmit} style={{
        background: "#fff", padding: "32px", borderRadius: "12px",
        boxShadow: "0 2px 12px rgba(0,0,0,0.1)", textAlign: "center", width: "320px"
      }}>
        <h2 style={{ marginBottom: "16px", color: "#2c3e9e" }}>Restricted Access</h2>

        <input
          type="text"
          placeholder="Email"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          style={{
            width: "100%", padding: "10px", marginBottom: "10px",
            borderRadius: "6px", border: "1px solid #ccc", boxSizing: "border-box"
          }}
        />

        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={{
            width: "100%", padding: "10px", marginBottom: "12px",
            borderRadius: "6px", border: "1px solid #ccc", boxSizing: "border-box"
          }}
        />

        <button type="submit" style={{
          width: "100%", padding: "10px", borderRadius: "6px",
          background: "#2c3e9e", color: "#fff", border: "none", cursor: "pointer"
        }}>
          Enter
        </button>

        {error && <p style={{ color: "red", marginTop: "10px" }}>Invalid email or password</p>}
      </form>
    </div>
  );
}
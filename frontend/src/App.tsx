import { useEffect, useState } from "react";

// M1: empty shell that confirms the backend is reachable.
// M2+ adds GPX drop, parameter controls, Three.js preview, and downloads.
export function App() {
  const [health, setHealth] = useState<string>("…");

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setHealth(d.status ?? "?"))
      .catch(() => setHealth("unreachable"));
  }, []);

  return (
    <main
      style={{
        fontFamily: "system-ui, sans-serif",
        maxWidth: 720,
        margin: "4rem auto",
        padding: "0 1rem",
        lineHeight: 1.6,
      }}
    >
      <h1>3d-footprint</h1>
      <p>GPXの移動軌跡＋地形＋建物から、3Dプリント用の 3MF / STL を生成します。</p>
      <p>
        backend: <strong>{health}</strong>
      </p>
      <p style={{ color: "#666" }}>M1 雛形。M2 以降で GPX 取込・地形生成・プレビューを追加します。</p>
    </main>
  );
}

import { useEffect, useRef, useState } from "react";

// M2: upload GPX -> backend builds the terrain solid -> download STL/3MF.
// Vertical exaggeration is adjustable here (the slider the print really needs).
export function App() {
  const [health, setHealth] = useState("…");
  const [file, setFile] = useState<File | null>(null);
  const [sizeMm, setSizeMm] = useState(120);
  const [verticalScale, setVerticalScale] = useState(8);
  const [baseThickness, setBaseThickness] = useState(3);
  const [includeTrack, setIncludeTrack] = useState(true);
  const [trackWidth, setTrackWidth] = useState(1.2);
  const [trackHeight, setTrackHeight] = useState(1.5);
  const [fmt, setFmt] = useState("stl");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setHealth(d.status ?? "?"))
      .catch(() => setHealth("unreachable"));
  }, []);

  async function generate() {
    if (!file) {
      setStatus("GPXファイルを選択してください");
      return;
    }
    setBusy(true);
    setStatus("生成中… (DEM取得→メッシュ化)");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("size_mm", String(sizeMm));
      form.append("vertical_scale", String(verticalScale));
      form.append("base_thickness_mm", String(baseThickness));
      form.append("include_track", String(includeTrack));
      form.append("track_width_mm", String(trackWidth));
      form.append("track_height_mm", String(trackHeight));
      form.append("fmt", fmt);

      const resp = await fetch("/api/generate", { method: "POST", body: form });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail ?? `HTTP ${resp.status}`);
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `footprint.${fmt}`;
      a.click();
      URL.revokeObjectURL(url);
      setStatus("完了。ダウンロードしました。");
    } catch (e) {
      setStatus(`エラー: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  const row: React.CSSProperties = { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, margin: "0.75rem 0" };

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 560, margin: "3rem auto", padding: "0 1rem", lineHeight: 1.6 }}>
      <h1>3d-footprint</h1>
      <p style={{ color: "#666", marginTop: -8 }}>
        GPXの移動軌跡＋地形から 3Dプリント用 3MF/STL を生成（backend: <strong>{health}</strong>）
      </p>

      <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: "1rem 1.25rem" }}>
        <div style={row}>
          <label>GPXファイル</label>
          <input ref={inputRef} type="file" accept=".gpx" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </div>

        <div style={row}>
          <label>垂直強調 ×{verticalScale}</label>
          <input style={{ flex: 1 }} type="range" min={1} max={30} step={0.5}
            value={verticalScale} onChange={(e) => setVerticalScale(Number(e.target.value))} />
        </div>

        <div style={row}>
          <label>出力サイズ（最大辺 mm）</label>
          <input type="number" min={20} max={300} value={sizeMm} onChange={(e) => setSizeMm(Number(e.target.value))} style={{ width: 90 }} />
        </div>

        <div style={row}>
          <label>底面厚（mm）</label>
          <input type="number" min={0} max={20} step={0.5} value={baseThickness} onChange={(e) => setBaseThickness(Number(e.target.value))} style={{ width: 90 }} />
        </div>

        <hr style={{ border: 0, borderTop: "1px solid #eee", margin: "0.5rem 0" }} />

        <div style={row}>
          <label>軌跡を含める</label>
          <input type="checkbox" checked={includeTrack} onChange={(e) => setIncludeTrack(e.target.checked)} />
        </div>

        <div style={{ ...row, opacity: includeTrack ? 1 : 0.4 }}>
          <label>軌跡の幅（mm）</label>
          <input type="number" min={0.4} max={10} step={0.1} value={trackWidth} disabled={!includeTrack} onChange={(e) => setTrackWidth(Number(e.target.value))} style={{ width: 90 }} />
        </div>

        <div style={{ ...row, opacity: includeTrack ? 1 : 0.4 }}>
          <label>軌跡の高さ（mm）</label>
          <input type="number" min={0.2} max={10} step={0.1} value={trackHeight} disabled={!includeTrack} onChange={(e) => setTrackHeight(Number(e.target.value))} style={{ width: 90 }} />
        </div>

        <hr style={{ border: 0, borderTop: "1px solid #eee", margin: "0.5rem 0" }} />

        <div style={row}>
          <label>フォーマット</label>
          <select value={fmt} onChange={(e) => setFmt(e.target.value)}>
            <option value="stl">STL（単色）</option>
            <option value="3mf">3MF</option>
          </select>
        </div>

        <button onClick={generate} disabled={busy} style={{ width: "100%", padding: "0.6rem", marginTop: "0.5rem", fontSize: 16, cursor: busy ? "wait" : "pointer" }}>
          {busy ? "生成中…" : "生成してダウンロード"}
        </button>
        {status && <p style={{ margin: "0.75rem 0 0", color: status.startsWith("エラー") ? "#c00" : "#333" }}>{status}</p>}
      </div>

      <p style={{ color: "#999", fontSize: 13, marginTop: "1rem" }}>
        M3: 地形＋軌跡の凸ライン。3MFは地形/軌跡を別ボディで出力（多色化の土台）。次はGLBプレビューと多色(M4)。
      </p>
    </main>
  );
}

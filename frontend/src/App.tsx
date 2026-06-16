import { useCallback, useEffect, useState } from "react";
import { Preview } from "./Preview";

// M4: live GLB preview (Three.js) + per-layer colors. Vertical scale defaults
// to x1; preview regenerates server-side (debounced) so the track height stays
// correct while only the terrain relief scales.
export function App() {
  const [health, setHealth] = useState("…");
  const [file, setFile] = useState<File | null>(null);
  const [sizeMm, setSizeMm] = useState(120);
  const [verticalScale, setVerticalScale] = useState(1);
  const [baseThickness, setBaseThickness] = useState(3);
  const [gridMax, setGridMax] = useState(1000);
  const [landuse, setLanduse] = useState(false);
  const [landuseSmooth, setLanduseSmooth] = useState(60);
  const [includeTrack, setIncludeTrack] = useState(true);
  const [trackWidth, setTrackWidth] = useState(1.2);
  const [trackHeight, setTrackHeight] = useState(1.5);
  const [includeBuildings, setIncludeBuildings] = useState(false);
  const [buildingScale, setBuildingScale] = useState(1);
  const [terrainColor, setTerrainColor] = useState("#c2b280");
  const [trackColor, setTrackColor] = useState("#dc4628");
  const [buildingColor, setBuildingColor] = useState("#b0b0b0");
  const [fmt, setFmt] = useState("3mf");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [glb, setGlb] = useState<ArrayBuffer | null>(null);

  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then((d) => setHealth(d.status ?? "?")).catch(() => setHealth("unreachable"));
  }, []);

  const buildForm = useCallback(
    (outFmt: string) => {
      const f = new FormData();
      f.append("file", file!);
      f.append("size_mm", String(sizeMm));
      f.append("vertical_scale", String(verticalScale));
      f.append("base_thickness_mm", String(baseThickness));
      f.append("grid_max", String(gridMax));
      f.append("landuse", String(landuse));
      f.append("landuse_smooth_m", String(landuseSmooth));
      f.append("include_track", String(includeTrack));
      f.append("track_width_mm", String(trackWidth));
      f.append("track_height_mm", String(trackHeight));
      f.append("include_buildings", String(includeBuildings));
      f.append("building_scale", String(buildingScale));
      f.append("terrain_color", terrainColor);
      f.append("track_color", trackColor);
      f.append("building_color", buildingColor);
      f.append("fmt", outFmt);
      return f;
    },
    [file, sizeMm, verticalScale, baseThickness, gridMax, landuse, landuseSmooth, includeTrack, trackWidth, trackHeight, includeBuildings, buildingScale, terrainColor, trackColor, buildingColor]
  );

  const LANDUSE_LEGEND: [string, string][] = [
    ["水域", "#4a80c0"], ["森林", "#3f7d3a"], ["農地", "#c9d17a"],
    ["建物用地", "#b0b0b0"], ["道路", "#6f6f6f"], ["荒地・海浜", "#cdbb8f"],
  ];

  // Debounced live preview whenever inputs change.
  useEffect(() => {
    if (!file) return;
    const id = setTimeout(async () => {
      setBusy(true);
      setStatus("プレビュー更新中…");
      try {
        const resp = await fetch("/api/generate", { method: "POST", body: buildForm("glb") });
        if (!resp.ok) throw new Error(((await resp.json().catch(() => ({}))) as any).detail ?? `HTTP ${resp.status}`);
        setGlb(await resp.arrayBuffer());
        setStatus("");
      } catch (e) {
        setStatus(`エラー: ${(e as Error).message}`);
      } finally {
        setBusy(false);
      }
    }, 500);
    return () => clearTimeout(id);
  }, [file, buildForm]);

  async function download() {
    if (!file) {
      setStatus("GPXファイルを選択してください");
      return;
    }
    setBusy(true);
    setStatus("生成中…");
    try {
      const resp = await fetch("/api/generate", { method: "POST", body: buildForm(fmt) });
      if (!resp.ok) throw new Error(((await resp.json().catch(() => ({}))) as any).detail ?? `HTTP ${resp.status}`);
      const ext = fmt === "stl_multi" ? "zip" : fmt;
      const url = URL.createObjectURL(await resp.blob());
      const a = document.createElement("a");
      a.href = url;
      a.download = `footprint.${ext}`;
      a.click();
      URL.revokeObjectURL(url);
      setStatus("ダウンロードしました。");
    } catch (e) {
      setStatus(`エラー: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  const row: React.CSSProperties = { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, margin: "0.6rem 0" };

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 980, margin: "2rem auto", padding: "0 1rem", lineHeight: 1.5 }}>
      <h1 style={{ marginBottom: 0 }}>3d-footprint</h1>
      <p style={{ color: "#666", marginTop: 4 }}>GPXの移動軌跡＋地形 → 3Dプリント用 3MF/STL（backend: <strong>{health}</strong>）</p>

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 20, alignItems: "start" }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: "0.75rem 1rem" }}>
          <div style={row}>
            <label>GPXファイル</label>
            <input type="file" accept=".gpx" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>

          <div style={row}>
            <label>垂直強調 ×{verticalScale}</label>
            <input style={{ flex: 1 }} type="range" min={1} max={30} step={0.5} value={verticalScale} onChange={(e) => setVerticalScale(Number(e.target.value))} />
          </div>
          <div style={row}>
            <label>出力サイズ（最大辺 mm）</label>
            <input type="number" min={20} max={300} value={sizeMm} onChange={(e) => setSizeMm(Number(e.target.value))} style={{ width: 80 }} />
          </div>
          <div style={row}>
            <label>底面厚（mm）</label>
            <input type="number" min={0} max={20} step={0.5} value={baseThickness} onChange={(e) => setBaseThickness(Number(e.target.value))} style={{ width: 80 }} />
          </div>
          <div style={row}>
            <label>解像度（詳細度）</label>
            <select value={gridMax} onChange={(e) => setGridMax(Number(e.target.value))}>
              <option value={700}>標準（速い・粗い）</option>
              <option value={1000}>高</option>
              <option value={1400}>最高（細かい・重い）</option>
            </select>
          </div>
          <div style={{ fontSize: 11, color: "#888", margin: "-0.3rem 0 0.4rem" }}>
            上げるほど色分け・地形が細かく（建物/橋に近づく）なりますが、生成とプレビューが重くなります。
          </div>
          <div style={row}>
            <label>土地利用で色分け</label>
            <input type="checkbox" checked={landuse} onChange={(e) => setLanduse(e.target.checked)} />
          </div>
          {landuse ? (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, fontSize: 12, color: "#555", margin: "0 0 0.4rem" }}>
              {LANDUSE_LEGEND.map(([name, c]) => (
                <span key={name} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 11, height: 11, background: c, borderRadius: 2, display: "inline-block" }} />
                  {name}
                </span>
              ))}
            </div>
          ) : (
            <div style={row}>
              <label>地形の色</label>
              <input type="color" value={terrainColor} onChange={(e) => setTerrainColor(e.target.value)} />
            </div>
          )}
          {landuse && (
            <>
              <div style={row}>
                <label>色分けのなめらかさ {landuseSmooth}m</label>
                <input style={{ flex: 1 }} type="range" min={0} max={200} step={10} value={landuseSmooth} onChange={(e) => setLanduseSmooth(Number(e.target.value))} />
              </div>
              <div style={{ fontSize: 11, color: "#888", margin: "-0.3rem 0 0.4rem" }}>
                0で補完オフ（生のマス目）。上げるほど境界を曲線化＋色を統合（PLATEAU域は自動で控えめ）。
              </div>
            </>
          )}

          <hr style={{ border: 0, borderTop: "1px solid #eee" }} />

          <div style={row}>
            <label>軌跡を含める</label>
            <input type="checkbox" checked={includeTrack} onChange={(e) => setIncludeTrack(e.target.checked)} />
          </div>
          <div style={{ ...row, opacity: includeTrack ? 1 : 0.4 }}>
            <label>軌跡の幅（mm）</label>
            <input type="number" min={0.4} max={10} step={0.1} value={trackWidth} disabled={!includeTrack} onChange={(e) => setTrackWidth(Number(e.target.value))} style={{ width: 80 }} />
          </div>
          <div style={{ ...row, opacity: includeTrack ? 1 : 0.4 }}>
            <label>軌跡の高さ（mm）</label>
            <input type="number" min={0.2} max={10} step={0.1} value={trackHeight} disabled={!includeTrack} onChange={(e) => setTrackHeight(Number(e.target.value))} style={{ width: 80 }} />
          </div>
          <div style={{ ...row, opacity: includeTrack ? 1 : 0.4 }}>
            <label>軌跡の色</label>
            <input type="color" value={trackColor} disabled={!includeTrack} onChange={(e) => setTrackColor(e.target.value)} />
          </div>

          <hr style={{ border: 0, borderTop: "1px solid #eee" }} />

          <div style={row}>
            <label>建物・橋 (PLATEAU LOD2)</label>
            <input type="checkbox" checked={includeBuildings} onChange={(e) => setIncludeBuildings(e.target.checked)} />
          </div>
          {includeBuildings && (
            <>
              <div style={{ fontSize: 12, color: "#888", margin: "0 0 0.4rem" }}>
                PLATEAU整備済みの都市のみ（LOD2/LOD1）。橋・高架も同じ色で含みます（実標高に配置）。初回はDLに時間がかかります。
              </div>
              <div style={row}>
                <label>建物の高さ強調 ×{buildingScale}</label>
                <input style={{ flex: 1 }} type="range" min={1} max={50} step={1} value={buildingScale} onChange={(e) => setBuildingScale(Number(e.target.value))} />
              </div>
              <div style={row}>
                <label>建物・橋の色</label>
                <input type="color" value={buildingColor} onChange={(e) => setBuildingColor(e.target.value)} />
              </div>
            </>
          )}

          <hr style={{ border: 0, borderTop: "1px solid #eee" }} />

          <div style={row}>
            <label>フォーマット</label>
            <select value={fmt} onChange={(e) => setFmt(e.target.value)}>
              <option value="3mf">3MF（多色・単一ファイル）</option>
              <option value="stl_multi">STL（多色・色ごと分割ZIP）</option>
              <option value="stl">STL（単色）</option>
            </select>
          </div>
          <button onClick={download} disabled={busy || !file} style={{ width: "100%", padding: "0.6rem", marginTop: 4, fontSize: 16, cursor: busy ? "wait" : "pointer" }}>
            生成してダウンロード
          </button>
          {status && <p style={{ margin: "0.6rem 0 0", color: status.startsWith("エラー") ? "#c00" : "#555", fontSize: 13 }}>{status}</p>}
        </div>

        <div style={{ position: "relative", border: "1px solid #ddd", borderRadius: 10, overflow: "hidden", height: 540, background: "#f0f0f0" }}>
          {!file && <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "#999" }}>GPXを選択するとプレビュー</div>}
          {busy && <div style={{ position: "absolute", top: 10, right: 12, background: "#000a", color: "#fff", padding: "2px 10px", borderRadius: 6, fontSize: 12 }}>更新中…</div>}
          <Preview glb={glb} />
        </div>
      </div>
    </main>
  );
}

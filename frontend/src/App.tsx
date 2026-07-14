import { useCallback, useEffect, useState } from "react";
import { MapPicker, type Bbox } from "./MapPicker";
import { Preview } from "./Preview";
import "./ui.css";

// Flow: pick a GPX -> the map shows the track and a draggable model bbox ->
// tune every option -> 「3Dモデルを作成する」 generates the GLB preview
// (server-side, so track height etc. stay exact) -> download as 3MF/STL.
export function App() {
  const [health, setHealth] = useState("…");
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [sizeMm, setSizeMm] = useState(120);
  const [verticalScale, setVerticalScale] = useState(1);
  const [baseThickness, setBaseThickness] = useState(3);
  const [gridMax, setGridMax] = useState(1000);
  const [landuse, setLanduse] = useState(false);
  const [includeTrack, setIncludeTrack] = useState(true);
  const [trackWidth, setTrackWidth] = useState(1.2);
  const [trackHeight, setTrackHeight] = useState(1.5);
  const [includeBuildings, setIncludeBuildings] = useState(false);
  const [buildingScale, setBuildingScale] = useState(1);
  const [minFeature, setMinFeature] = useState(0.8);
  const [terrainColor, setTerrainColor] = useState("#c2b280");
  const [trackColor, setTrackColor] = useState("#dc4628");
  const [buildingColor, setBuildingColor] = useState("#b0b0b0");
  const [fmt, setFmt] = useState("3mf");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [glb, setGlb] = useState<ArrayBuffer | null>(null);
  const [bbox, setBbox] = useState<Bbox | null>(null);
  const [autoBbox, setAutoBbox] = useState<Bbox | null>(null);
  const [trackPts, setTrackPts] = useState<[number, number][]>([]);

  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then((d) => setHealth(d.status ?? "?")).catch(() => setHealth("unreachable"));
  }, []);

  // Parse the GPX in the browser: track polyline for the map, plus the same
  // automatic extent the backend would use (track bbox + 8% margin) as the
  // initial model bbox.
  useEffect(() => {
    if (!file) { setAutoBbox(null); setBbox(null); setTrackPts([]); return; }
    let stale = false;
    file.text().then((text) => {
      if (stale) return;
      const doc = new DOMParser().parseFromString(text, "application/xml");
      let els = doc.getElementsByTagNameNS("*", "trkpt");
      if (!els.length) els = doc.getElementsByTagNameNS("*", "rtept");
      if (!els.length) els = doc.getElementsByTagNameNS("*", "wpt");
      const pts: [number, number][] = [];
      for (const p of Array.from(els)) {
        const lon = Number(p.getAttribute("lon"));
        const lat = Number(p.getAttribute("lat"));
        if (Number.isFinite(lon) && Number.isFinite(lat)) pts.push([lat, lon]);
      }
      if (!pts.length) { setAutoBbox(null); setBbox(null); setTrackPts([]); return; }
      let lo0 = Infinity, lo1 = -Infinity, la0 = Infinity, la1 = -Infinity;
      for (const [lat, lon] of pts) {
        lo0 = Math.min(lo0, lon); lo1 = Math.max(lo1, lon);
        la0 = Math.min(la0, lat); la1 = Math.max(la1, lat);
      }
      const dlon = Math.max(lo1 - lo0, 1e-3) * 0.08;
      const dlat = Math.max(la1 - la0, 1e-3) * 0.08;
      const bb: Bbox = [lo0 - dlon, la0 - dlat, lo1 + dlon, la1 + dlat];
      // Cap the polyline so huge 1 Hz logs don't bog the map down.
      const stride = Math.max(1, Math.ceil(pts.length / 3000));
      setTrackPts(pts.filter((_, i) => i % stride === 0 || i === pts.length - 1));
      setAutoBbox(bb);
      setBbox(bb);
    });
    return () => { stale = true; };
  }, [file]);

  const bboxParam = bbox ? bbox.map((v) => v.toFixed(6)).join(",") : "";

  const buildForm = useCallback(
    (outFmt: string) => {
      const f = new FormData();
      f.append("file", file!);
      f.append("size_mm", String(sizeMm));
      f.append("vertical_scale", String(verticalScale));
      f.append("base_thickness_mm", String(baseThickness));
      f.append("grid_max", String(gridMax));
      f.append("landuse", String(landuse));
      f.append("include_track", String(includeTrack));
      f.append("track_width_mm", String(trackWidth));
      f.append("track_height_mm", String(trackHeight));
      f.append("include_buildings", String(includeBuildings));
      f.append("building_scale", String(buildingScale));
      f.append("min_feature_mm", String(minFeature));
      f.append("terrain_color", terrainColor);
      f.append("track_color", trackColor);
      f.append("building_color", buildingColor);
      if (bboxParam) f.append("bbox", bboxParam);
      f.append("fmt", outFmt);
      return f;
    },
    [file, sizeMm, verticalScale, baseThickness, gridMax, landuse, includeTrack, trackWidth, trackHeight, includeBuildings, buildingScale, minFeature, terrainColor, trackColor, buildingColor, bboxParam]
  );

  // PLATEAU 土地利用（luse）区分 → 印刷カテゴリ。backend/app/core/coloring.py と対応。
  const LANDUSE_LEGEND: [string, string][] = [
    ["水面", "#4a80c0"], ["森林・緑地", "#3f7d3a"], ["農地", "#c9d17a"],
    ["市街地", "#b0b0b0"], ["道路", "#6f6f6f"], ["空地・荒地", "#cdbb8f"],
  ];

  // Enforce the backend's minimum span (0.001 deg per side) as the rectangle
  // is dragged, expanding around the centre if the user makes it too small.
  const onBboxChange = useCallback((bb: Bbox) => {
    let [w, s, e, n] = bb;
    if (e - w < 1e-3) { const c = (w + e) / 2; w = c - 5e-4; e = c + 5e-4; }
    if (n - s < 1e-3) { const c = (s + n) / 2; s = c - 5e-4; n = c + 5e-4; }
    setBbox([w, s, e, n]);
  }, []);

  // Generation only runs on the button, not on every tweak.
  async function createModel() {
    if (!file) {
      setStatus("GPXファイルを選択してください");
      return;
    }
    setBusy(true);
    setStatus("3Dモデル生成中…");
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
  }

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

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  }

  return (
    <main className="app">
      <header className="app-header">
        <h1 className="app-title"><span className="logo">🗻</span>3d-footprint</h1>
        <p className="app-subtitle">GPXの移動軌跡＋地形 → 3Dプリント用 3MF/STL</p>
        <span className={`health${health === "ok" ? " ok" : ""}`}>
          <span className="dot" />API {health === "ok" ? "接続中" : health}
        </span>
      </header>

      <div className="layout">
        <div className="card panel">
          <label
            className={`dropzone${dragOver ? " drag" : ""}${file ? " has-file" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            <input type="file" accept=".gpx" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            {file ? (
              <>
                <div className="file-name">📍 {file.name}</div>
                <div className="sub">クリックまたはドロップで差し替え</div>
              </>
            ) : (
              <>
                <div>GPXファイルをここにドロップ</div>
                <div className="sub">またはクリックして選択</div>
              </>
            )}
          </label>

          <h3 className="section-title">モデル</h3>
          <div className="row">
            <label>垂直強調<span className="val">×{verticalScale}</span></label>
            <input type="range" min={1} max={30} step={0.5} value={verticalScale} onChange={(e) => setVerticalScale(Number(e.target.value))} />
          </div>
          <div className="row">
            <label>出力サイズ（最大辺 mm）</label>
            <input type="number" min={20} max={300} value={sizeMm} onChange={(e) => setSizeMm(Number(e.target.value))} />
          </div>
          <div className="row">
            <label>底面厚（mm）</label>
            <input type="number" min={0} max={20} step={0.5} value={baseThickness} onChange={(e) => setBaseThickness(Number(e.target.value))} />
          </div>
          <div className="row">
            <label>解像度（詳細度）</label>
            <select value={gridMax} onChange={(e) => setGridMax(Number(e.target.value))}>
              <option value={700}>標準（速い・粗い）</option>
              <option value={1000}>高</option>
              <option value={1400}>最高（細かい・重い）</option>
            </select>
          </div>
          <p className="hint">
            上げるほど色分け・地形が細かく（建物/橋に近づく）なりますが、生成とプレビューが重くなります。
          </p>

          <h3 className="section-title">色・土地利用</h3>
          <div className="row">
            <label>土地利用で色分け</label>
            <input className="toggle" type="checkbox" checked={landuse} onChange={(e) => setLanduse(e.target.checked)} />
          </div>
          {landuse && (
            <>
              <div className="legend">
                {LANDUSE_LEGEND.map(([name, c]) => (
                  <span key={name}><i style={{ background: c }} />{name}</span>
                ))}
              </div>
              <p className="hint">
                PLATEAU（土地利用）を優先し、無い部分はJAXA土地被覆図（10m）で補完。どちらにも無い部分は「地形の色」になります（道路はPLATEAU域のみ）。
              </p>
            </>
          )}
          <div className="row">
            <label>地形の色</label>
            <input type="color" value={terrainColor} onChange={(e) => setTerrainColor(e.target.value)} />
          </div>

          <h3 className="section-title">軌跡</h3>
          <div className="row">
            <label>軌跡を含める</label>
            <input className="toggle" type="checkbox" checked={includeTrack} onChange={(e) => setIncludeTrack(e.target.checked)} />
          </div>
          <div className={`row${includeTrack ? "" : " dim"}`}>
            <label>軌跡の幅（mm）</label>
            <input type="number" min={0.4} max={10} step={0.1} value={trackWidth} disabled={!includeTrack} onChange={(e) => setTrackWidth(Number(e.target.value))} />
          </div>
          <div className={`row${includeTrack ? "" : " dim"}`}>
            <label>軌跡の高さ（mm）</label>
            <input type="number" min={0.2} max={10} step={0.1} value={trackHeight} disabled={!includeTrack} onChange={(e) => setTrackHeight(Number(e.target.value))} />
          </div>
          <div className={`row${includeTrack ? "" : " dim"}`}>
            <label>軌跡の色</label>
            <input type="color" value={trackColor} disabled={!includeTrack} onChange={(e) => setTrackColor(e.target.value)} />
          </div>

          <h3 className="section-title">建物・橋</h3>
          <div className="row">
            <label>建物・橋 (PLATEAU LOD2)</label>
            <input className="toggle" type="checkbox" checked={includeBuildings} onChange={(e) => setIncludeBuildings(e.target.checked)} />
          </div>
          {includeBuildings && (
            <>
              <p className="hint">
                PLATEAU整備済みの都市のみ（LOD2/LOD1）。印刷用に簡略化（建物＝輪郭ブロック化／橋・高架＝デッキ＋脚で地面に接続）。初回はDLに時間がかかります。
              </p>
              <div className="row">
                <label>高さ強調<span className="val">×{buildingScale}</span></label>
                <input type="range" min={1} max={50} step={1} value={buildingScale} onChange={(e) => setBuildingScale(Number(e.target.value))} />
              </div>
              <div className="row">
                <label>最小幅<span className="val">{minFeature}mm</span></label>
                <input type="range" min={0.4} max={2} step={0.1} value={minFeature} onChange={(e) => setMinFeature(Number(e.target.value))} />
              </div>
              <p className="hint">
                ノズル径以下は潰れるため、これより細い建物・橋脚は最小幅まで太らせます（目安: ノズル0.4mmなら0.8）。
              </p>
              <div className="row">
                <label>建物・橋の色</label>
                <input type="color" value={buildingColor} onChange={(e) => setBuildingColor(e.target.value)} />
              </div>
            </>
          )}

          <hr className="divider" />

          <button className="btn btn-primary btn-block" onClick={createModel} disabled={busy || !file}>
            {busy && <span className="spinner" />}
            3Dモデルを作成する
          </button>

          <div className="row" style={{ marginTop: 6 }}>
            <label>フォーマット</label>
            <select value={fmt} onChange={(e) => setFmt(e.target.value)}>
              <option value="3mf">3MF（多色・単一ファイル）</option>
              <option value="stl_multi">STL（多色・色ごと分割ZIP）</option>
              <option value="stl">STL（単色）</option>
            </select>
          </div>
          <button className="btn btn-secondary btn-block" onClick={download} disabled={busy || !file}>
            生成してダウンロード
          </button>
          {status && <p className={`status${status.startsWith("エラー") ? " error" : ""}`}>{status}</p>}
        </div>

        <div className="stack">
          <div className="card">
            <div className="card-head">
              <strong>モデル化する範囲</strong>
              <span>角の■をドラッグして調整</span>
              {bbox && (
                <span className="coords">
                  西{bbox[0].toFixed(4)} 南{bbox[1].toFixed(4)} 東{bbox[2].toFixed(4)} 北{bbox[3].toFixed(4)}
                </span>
              )}
              <span style={{ flex: 1 }} />
              <button className="btn btn-ghost" onClick={() => autoBbox && setBbox(autoBbox)} disabled={!autoBbox}>
                軌跡に合わせる
              </button>
            </div>
            <div className="card-body map-box">
              {!file && <div className="overlay-hint">GPXを選択すると地図に軌跡と範囲を表示</div>}
              <MapPicker points={trackPts} bbox={bbox} onBboxChange={onBboxChange} />
            </div>
          </div>

          <div className="card">
            <div className="card-head">
              <strong>3Dプレビュー</strong>
              <span>ドラッグで回転・ホイールで拡大</span>
            </div>
            <div className="card-body preview-box">
              {!glb && <div className="overlay-hint">「3Dモデルを作成する」を押すとここにプレビュー</div>}
              {busy && <div className="busy-badge"><span className="spinner" />生成中…</div>}
              <Preview glb={glb} />
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}

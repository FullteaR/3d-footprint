import { useCallback, useEffect, useRef, useState } from "react";
import {
  MapPicker, extentMeters, fitBbox, normalizeBbox, scaleBbox, spanMeters,
  type Bbox, type Shape,
} from "./MapPicker";
import { Preview } from "./Preview";
import "./ui.css";

// 範囲 (real-world span) / 印刷サイズ / 縮尺 are one equation apart
// (span_m = size_mm/1000 × denom): the locked one is frozen, editing either
// of the others makes the remaining one follow.
type Lock = "span" | "size" | "scale";

// Number input that commits on blur / Enter — committed values can move the
// map bbox, so per-keystroke commits (2 → 25 → 250…) would thrash the frame.
function NumField({ value, min, max, step, digits = 0, disabled, onCommit }: {
  value: number | null;
  min: number;
  max: number;
  step?: number;
  digits?: number;
  disabled?: boolean;
  onCommit: (v: number) => void;
}) {
  const fmt = (v: number | null) => (v == null ? "" : Number(v.toFixed(digits)).toString());
  const [txt, setTxt] = useState(fmt(value));
  useEffect(() => { setTxt(fmt(value)); }, [value]); // resync when the value is driven from elsewhere
  const commit = () => {
    const v = Number(txt);
    if (txt.trim() !== "" && Number.isFinite(v)) {
      const cl = Math.min(max, Math.max(min, v));
      setTxt(fmt(cl));
      onCommit(cl);
    } else {
      setTxt(fmt(value));
    }
  };
  return (
    <input
      type="number" value={txt} min={min} max={max} step={step} disabled={disabled}
      onChange={(e) => setTxt(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
    />
  );
}

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
  const [includePlate, setIncludePlate] = useState(false);
  const [plateSvg, setPlateSvg] = useState<File | null>(null);
  const [plateUrl, setPlateUrl] = useState<string | null>(null);
  const [plateDepth, setPlateDepth] = useState(16);
  const [plateRelief, setPlateRelief] = useState(0.6);
  const [labelColor, setLabelColor] = useState("#333333");
  const [terrainColor, setTerrainColor] = useState("#c2b280");
  const [trackColor, setTrackColor] = useState("#dc4628");
  const [buildingColor, setBuildingColor] = useState("#b0b0b0");
  const [fmt, setFmt] = useState("3mf");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [glb, setGlb] = useState<ArrayBuffer | null>(null);
  const [bbox, setBbox] = useState<Bbox | null>(null);
  const [shape, setShape] = useState<Shape>("rect");
  const [rotation, setRotation] = useState(0);
  const [trackPts, setTrackPts] = useState<[number, number][]>([]);
  const [locked, setLocked] = useState<Lock>("size");
  const [lockedDenom, setLockedDenom] = useState<number | null>(null);
  // For the file-parse effect (which must not re-run on shape/rotation change)
  // and the Leaflet callbacks (created once).
  const shapeRef = useRef(shape); shapeRef.current = shape;
  const rotationRef = useRef(rotation); rotationRef.current = rotation;
  const bboxRef = useRef(bbox); bboxRef.current = bbox;
  const lockedRef = useRef(locked); lockedRef.current = locked;
  const lockedDenomRef = useRef(lockedDenom); lockedDenomRef.current = lockedDenom;

  const SIZE_MIN = 20, SIZE_MAX = 300;

  // Every bbox write goes through the lock: with the scale locked, the print
  // size follows the frame (clamped to the printable range, which also stops
  // an over-drag); the extent lock instead disables map resizing and reroutes
  // fit-to-track through recentre().
  const applyBbox = useCallback((raw: Bbox | null) => {
    if (!raw) { setBbox(null); return; }
    let bb = normalizeBbox(raw, shapeRef.current);
    const d = lockedDenomRef.current;
    if (lockedRef.current === "scale" && d) {
      let size = (spanMeters(bb, shapeRef.current) * 1000) / d;
      const cl = Math.min(SIZE_MAX, Math.max(SIZE_MIN, size));
      if (cl !== size) {
        bb = normalizeBbox(
          scaleBbox(bb, ((cl * d) / 1000) / spanMeters(bb, shapeRef.current)),
          shapeRef.current,
        );
        size = cl;
      }
      setSizeMm(size);
    }
    setBbox(bb);
  }, []);

  // Move the frame to `target`'s centre without resizing (for the extent lock).
  const recentre = useCallback((target: Bbox) => {
    const b = bboxRef.current;
    if (!b) { applyBbox(target); return; }
    const dlon = (target[0] + target[2]) / 2 - (b[0] + b[2]) / 2;
    const dlat = (target[1] + target[3]) / 2 - (b[1] + b[3]) / 2;
    setBbox([b[0] + dlon, b[1] + dlat, b[2] + dlon, b[3] + dlat]);
  }, [applyBbox]);

  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then((d) => setHealth(d.status ?? "?")).catch(() => setHealth("unreachable"));
  }, []);

  // Parse the GPX in the browser: track polyline for the map, plus the
  // automatic extent (track + 8% margin, in the current shape/rotation) as the
  // initial model outline.
  useEffect(() => {
    if (!file) { setBbox(null); setTrackPts([]); return; }
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
      if (!pts.length) { setBbox(null); setTrackPts([]); return; }
      // Cap the polyline so huge 1 Hz logs don't bog the map down.
      const stride = Math.max(1, Math.ceil(pts.length / 3000));
      setTrackPts(pts.filter((_, i) => i % stride === 0 || i === pts.length - 1));
      const fit = fitBbox(pts, shapeRef.current, rotationRef.current);
      if (fit && lockedRef.current === "span" && bboxRef.current) recentre(fit);
      else applyBbox(fit);
    });
    return () => { stale = true; };
  }, [file, applyBbox, recentre]);

  // Object URL for the nameplate SVG preview.
  useEffect(() => {
    if (!plateSvg) { setPlateUrl(null); return; }
    const u = URL.createObjectURL(plateSvg);
    setPlateUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [plateSvg]);

  const bboxParam = bbox ? bbox.map((v) => v.toFixed(6)).join(",") : "";

  const spanM = bbox ? spanMeters(bbox, shape) : null;
  // While the scale is locked it is the anchor; otherwise it's derived.
  const scaleDenom =
    locked === "scale" && lockedDenom ? lockedDenom
    : spanM ? (spanM * 1000) / sizeMm : null;

  function lockTo(item: Lock) {
    if (item === "scale") {
      if (scaleDenom == null) return;
      setLockedDenom(scaleDenom);
    }
    setLocked(item);
  }

  function commitSize(v: number) {
    setSizeMm(v);
    const b = bboxRef.current;
    if (locked === "scale" && lockedDenom && b) {
      const k = ((v / 1000) * lockedDenom) / spanMeters(b, shape);
      setBbox(normalizeBbox(scaleBbox(b, k), shape));
    }
  }

  function commitScale(d: number) {
    const b = bboxRef.current;
    if (!b) return;
    if (locked === "size") {
      const k = ((d * sizeMm) / 1000) / spanMeters(b, shape);
      setBbox(normalizeBbox(scaleBbox(b, k), shape));
    } else if (locked === "span") {
      setSizeMm(Math.min(SIZE_MAX, Math.max(SIZE_MIN, (spanMeters(b, shape) * 1000) / d)));
    }
  }

  const extentText = (() => {
    if (!bbox) return "—";
    const [w, h] = extentMeters(bbox);
    const f = (m: number) => (m >= 1000 ? `${(m / 1000).toFixed(2)}km` : `${Math.round(m)}m`);
    return `${f(w)} × ${f(h)}`;
  })();

  // Printed width of the nameplate slab (the model's front edge; a hexagon's
  // flat bottom edge is its middle half) — for the preview's aspect ratio.
  const plateWmm = (() => {
    if (!bbox || !spanM) return sizeMm;
    const mm = (sizeMm * extentMeters(bbox)[0]) / spanM;
    return shape === "hex" ? mm / 2 : mm;
  })();

  const lockBtn = (item: Lock, disabled = false) => (
    <button
      className={`lock-btn${locked === item ? " on" : ""}`}
      title={locked === item ? "固定中" : "この項目を固定する"}
      disabled={disabled}
      onClick={() => lockTo(item)}
    >
      {locked === item ? "🔒" : "🔓"}
    </button>
  );

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
      if (includePlate && plateSvg) {
        f.append("plate_svg", plateSvg);
        f.append("plate_depth_mm", String(plateDepth));
        f.append("plate_relief_mm", String(plateRelief));
        f.append("label_color", labelColor);
      }
      if (bboxParam) f.append("bbox", bboxParam);
      f.append("shape", shape);
      f.append("rotation_deg", String(rotation));
      f.append("fmt", outFmt);
      return f;
    },
    [file, sizeMm, verticalScale, baseThickness, gridMax, landuse, includeTrack, trackWidth, trackHeight, includeBuildings, buildingScale, minFeature, terrainColor, trackColor, buildingColor, includePlate, plateSvg, plateDepth, plateRelief, labelColor, bboxParam, shape, rotation]
  );

  // PLATEAU 土地利用（luse）区分 → 印刷カテゴリ。backend/app/core/coloring.py と対応。
  const LANDUSE_LEGEND: [string, string][] = [
    ["水面", "#4a80c0"], ["森林・緑地", "#3f7d3a"], ["農地", "#c9d17a"],
    ["市街地", "#b0b0b0"], ["道路", "#6f6f6f"], ["空地・荒地", "#cdbb8f"],
  ];

  // Minimum span + shape aspect ratio + the lock are enforced centrally so
  // every source (drag, shape switch, fit-to-track) yields a valid bbox.
  const onBboxChange = useCallback((bb: Bbox) => { applyBbox(bb); }, [applyBbox]);

  const onShapeChange = useCallback((sh: Shape) => {
    shapeRef.current = sh; // applyBbox must see the new shape before the render
    setShape(sh);
    if (bboxRef.current) applyBbox(bboxRef.current);
  }, [applyBbox]);

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

          <h3 className="section-title">大きさ・縮尺</h3>
          <div className="row">
            {lockBtn("span", !bbox)}
            <label>範囲（実距離）</label>
            <span className="val">{extentText}</span>
          </div>
          <div className="row">
            {lockBtn("size")}
            <label>印刷サイズ（最大辺 mm）</label>
            <NumField value={sizeMm} min={SIZE_MIN} max={SIZE_MAX} step={5} digits={1} disabled={locked === "size"} onCommit={commitSize} />
          </div>
          <div className="row">
            {lockBtn("scale", !bbox)}
            <label>縮尺　1:</label>
            <NumField value={scaleDenom == null ? null : Math.round(scaleDenom)} min={100} max={10000000} step={1000} disabled={!bbox || locked === "scale"} onCommit={commitScale} />
          </div>
          <div className="row presets">
            {[25000, 50000].map((d) => (
              <button key={d} className="btn btn-ghost btn-xs" disabled={!bbox || locked === "scale"} onClick={() => commitScale(d)}>
                1:{d.toLocaleString()}
              </button>
            ))}
          </div>
          <p className="hint">
            🔒の項目は固定。他の項目を動かすと、残りの項目が追従します。
            範囲を固定すると地図の枠はリサイズ不可（移動・回転は可）になります。
            縮尺を固定すると、枠のドラッグに印刷サイズが追従します。
          </p>

          <h3 className="section-title">モデル</h3>
          <div className="row">
            <label>垂直強調<span className="val">×{verticalScale}</span></label>
            <input type="range" min={1} max={30} step={0.5} value={verticalScale} onChange={(e) => setVerticalScale(Number(e.target.value))} />
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

          <h3 className="section-title">銘板</h3>
          <div className="row">
            <label>銘板を付ける</label>
            <input className="toggle" type="checkbox" checked={includePlate} onChange={(e) => setIncludePlate(e.target.checked)} />
          </div>
          {includePlate && (
            <>
              <label className={`dropzone slim${plateSvg ? " has-file" : ""}`}>
                <input type="file" accept=".svg,image/svg+xml" onChange={(e) => setPlateSvg(e.target.files?.[0] ?? null)} />
                {plateSvg ? (
                  <div className="file-name">🏷️ {plateSvg.name}</div>
                ) : (
                  <div>銘板デザインのSVGを選択</div>
                )}
                <div className="sub">板の面（約{Math.round(plateWmm)}×{plateDepth}mm）に自動で収めます</div>
              </label>
              {plateUrl && (
                <div className="plate-preview" style={{ aspectRatio: `${plateWmm} / ${plateDepth}`, background: terrainColor }}>
                  <img src={plateUrl} alt="銘板プレビュー" />
                </div>
              )}
              <p className="hint">
                モデル手前に張り出す板に、SVGの塗り・線がそのまま凸になります。
                文字はデザインツールで<b>アウトライン化</b>（パスに変換）してください（&lt;text&gt;要素は不可）。
                実寸0.4mm未満の細線は印刷で潰れます。縮尺や日付は上の縮尺表示を見てSVGに直接入れてください。
              </p>
              <div className="row">
                <label>板の奥行き（mm）</label>
                <input type="number" min={4} max={40} step={1} value={plateDepth} onChange={(e) => setPlateDepth(Number(e.target.value))} />
              </div>
              <div className="row">
                <label>凸の高さ（mm）</label>
                <input type="number" min={0.2} max={2} step={0.1} value={plateRelief} onChange={(e) => setPlateRelief(Number(e.target.value))} />
              </div>
              <div className="row">
                <label>凸部の色</label>
                <input type="color" value={labelColor} onChange={(e) => setLabelColor(e.target.value)} />
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
              <select value={shape} onChange={(e) => onShapeChange(e.target.value as Shape)}>
                <option value="rect">長方形</option>
                <option value="square">正方形</option>
                <option value="hex">正六角形</option>
              </select>
              <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                回転
                <input
                  type="number" step={5} value={rotation} style={{ width: 62 }}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (Number.isFinite(v)) setRotation(v);
                  }}
                />°
              </label>
              <span>{locked === "span" ? "●回転・✥移動（実距離を固定中）" : "■サイズ・●回転・✥移動"}</span>
              {bbox && (
                <span className="coords">
                  中心 {(((bbox[1] + bbox[3]) / 2)).toFixed(4)}, {(((bbox[0] + bbox[2]) / 2)).toFixed(4)}
                </span>
              )}
              <span style={{ flex: 1 }} />
              <button
                className="btn btn-ghost"
                onClick={() => {
                  const bb = fitBbox(trackPts, shape, rotation);
                  if (!bb) return;
                  if (locked === "span" && bbox) recentre(bb); else applyBbox(bb);
                }}
                disabled={!trackPts.length}
              >
                {locked === "span" ? "軌跡の中心へ" : "軌跡に合わせる"}
              </button>
            </div>
            <div className="card-body map-box">
              {!file && <div className="overlay-hint">GPXを選択すると地図に軌跡と範囲を表示</div>}
              <MapPicker
                points={trackPts} bbox={bbox} shape={shape} rotation={rotation}
                resizable={locked !== "span"}
                onBboxChange={onBboxChange} onRotationChange={setRotation}
              />
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

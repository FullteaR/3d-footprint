import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  MapPicker, clampPlate, extentMeters, fitBbox, freeSpot, normalizeBbox,
  scaleBbox, spanMeters, type Bbox, type PlateEdit, type Shape,
} from "./MapPicker";
import { Preview } from "./Preview";
import { TEXT, initialLang, localeOf, rememberLang, type Lang, type Text } from "./i18n";
import "./ui.css";

// 範囲 (real-world span) / 印刷サイズ / 縮尺 are one equation apart
// (span_m = size_mm/1000 × denom): the locked one is frozen, editing either
// of the others makes the remaining one follow. Nothing has to be locked
// (the default) — then the map frame is the master and the two numbers just
// describe it, trading off against each other when typed into.
type Lock = "span" | "size" | "scale";

// What the action bar is saying right now.
type Status =
  | { kind: "needFile" | "generating" | "downloading" | "downloaded" }
  | { kind: "error"; detail: string }
  | null;
const STATUS_TEXT = {
  needFile: "stNeedFile", generating: "stGenerating",
  downloading: "stDownloading", downloaded: "stDownloaded",
} as const;

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

// Two overlapping range inputs make one dual-thumb slider: only the thumbs
// take pointer events, so either one is grabbable even where they overlap.
function RangeSlider({ min, max, step, value, labels, onChange }: {
  min: number;
  max: number;
  step: number;
  value: [number, number];
  labels: [string, string];
  onChange: (v: [number, number]) => void;
}) {
  const [lo, hi] = value;
  const pct = (v: number) => (100 * (v - min)) / Math.max(max - min, 1e-9);
  return (
    <div className="range-dual">
      <span className="rail" />
      <span className="fill" style={{ left: `${pct(lo)}%`, right: `${100 - pct(hi)}%` }} />
      <input
        type="range" min={min} max={max} step={step} value={lo} aria-label={labels[0]}
        onChange={(e) => onChange([Math.min(Number(e.target.value), hi), hi])}
      />
      <input
        type="range" min={min} max={max} step={step} value={hi} aria-label={labels[1]}
        onChange={(e) => onChange([lo, Math.max(Number(e.target.value), lo)])}
      />
    </div>
  );
}

// GPX times are UTC; both are shown in the viewer's local time.
const pad2 = (n: number) => String(n).padStart(2, "0");
const clockText = (sec: number, withDate: boolean) => {
  const d = new Date(sec * 1000);
  const hm = `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
  return withDate ? `${d.getMonth() + 1}/${d.getDate()} ${hm}` : hm;
};

// A *starred* run in a dictionary string is the one word worth emphasising.
function rich(s: string) {
  return s.split("*").map((part, i) => (i % 2 ? <b key={i}>{part}</b> : part));
}

// Unclassified ground — what the land-use colours sit in, and the nameplate
// slab (mirrors terrain_color's backend default; no longer user-editable).
const TERRAIN_COLOR = "#c2b280";

// Printable range for a nameplate side (mm), for both the fields and the
// map's corner handles.
const PLATE_MIN_MM = 5, PLATE_MAX_MM = 200;

// PLATEAU 土地利用（luse）区分 → 印刷カテゴリ。backend/app/core/coloring.py と対応。
const legendOf = (t: Text): [string, string][] => [
  [t.luWater, "#4a80c0"], [t.luForest, "#3f7d3a"], [t.luFarm, "#c9d17a"],
  [t.luUrban, "#b0b0b0"], [t.luRoad, "#6f6f6f"], [t.luBare, "#cdbb8f"],
];

// Round map scales — the 1/1.5/2/2.5/3/4/5/7.5 ladder per decade, over the
// range the 縮尺 field accepts.
const SCALE_LADDER = [2, 3, 4, 5, 6].flatMap((k) =>
  [1, 1.5, 2, 2.5, 3, 4, 5, 7.5].map((m) => Math.round(m * 10 ** k)),
).filter((d) => d >= 100 && d <= 10000000);

// The three round scales nearest the current one (ratio-wise — scales are
// multiplicative), so the suggestions stay useful at any zoom.
function niceScales(denom: number): number[] {
  let best = 0;
  for (let i = 1; i < SCALE_LADDER.length; i++) {
    if (Math.abs(Math.log(SCALE_LADDER[i] / denom)) <
        Math.abs(Math.log(SCALE_LADDER[best] / denom))) best = i;
  }
  const from = Math.min(Math.max(best - 1, 0), SCALE_LADDER.length - 3);
  return SCALE_LADDER.slice(from, from + 3);
}

// Cap a polyline so huge 1 Hz logs don't bog the map down.
function thin(pts: [number, number][]): [number, number][] {
  const stride = Math.max(1, Math.ceil(pts.length / 3000));
  return pts.filter((_, i) => i % stride === 0 || i === pts.length - 1);
}

// Padlock for the span/size/scale lock buttons (emoji render too unevenly).
function LockIcon({ open }: { open: boolean }) {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
      {open ? (
        <path d="M11 7V4.5a3 3 0 0 0-6 0" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      ) : (
        <path d="M5 7V4.8a3 3 0 0 1 6 0V7" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      )}
      <rect x="3" y="7" width="10" height="6.5" rx="1.6" fill="currentColor" />
    </svg>
  );
}

// Nested contour rings with the track-vermilion summit dot.
function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 26 26" width="24" height="24" aria-hidden="true">
      <path d="M13 2.8C6.7 2.6 2.2 6.9 2.6 13.2c.4 6 5 10.3 10.9 10 6-.3 10-4.8 9.9-10.4C23.2 7 19.2 3 13 2.8Z" fill="none" stroke="#2e5e8c" strokeWidth="1.3" />
      <path d="M13.2 6.4c-4-.2-7 2.5-6.8 6.6.2 4 3.2 6.8 7 6.6 3.8-.2 6.3-3 6.2-6.7-.1-3.8-2.6-6.3-6.4-6.5Z" fill="none" stroke="#2f6b52" strokeWidth="1.3" />
      <path d="M13 10c-2-.1-3.6 1.3-3.5 3.3.1 2 1.6 3.4 3.6 3.3 1.9-.1 3.2-1.5 3.1-3.4-.1-1.9-1.3-3.1-3.2-3.2Z" fill="none" stroke="#c2b280" strokeWidth="1.3" />
      <circle cx="13.1" cy="13.3" r="1.7" fill="#f4552b" />
    </svg>
  );
}

// Flow: pick a GPX -> the map shows the track and a draggable model bbox ->
// tune every option -> 「3Dモデルを作成する」 generates the GLB preview
// (server-side, so track height etc. stay exact) -> download as 3MF/STL.
export function App() {
  const [lang, setLang] = useState<Lang>(initialLang);
  const t = TEXT[lang];
  const [health, setHealth] = useState("…");
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [sizeMm, setSizeMm] = useState(120);
  const [verticalScale, setVerticalScale] = useState(1);
  const [baseThickness, setBaseThickness] = useState(3);
  const [gridMax, setGridMax] = useState(1000);
  const [minColor, setMinColor] = useState(1);
  const [includeTrack, setIncludeTrack] = useState(true);
  const [trackWidth, setTrackWidth] = useState(1.2);
  const [trackHeight, setTrackHeight] = useState(1.5);
  const [includeBuildings, setIncludeBuildings] = useState(false);
  const [buildingScale, setBuildingScale] = useState(1);
  const [minFeature, setMinFeature] = useState(0.8);
  const [includePlate, setIncludePlate] = useState(false);
  const [plateSvg, setPlateSvg] = useState<File | null>(null);
  const [plateUrl, setPlateUrl] = useState<string | null>(null);
  // The plate is a pad laid on the map at plateCenter ([lat, lon]), sized in
  // printed mm and turned by plateRotation relative to the model outline.
  const [plateCenter, setPlateCenter] = useState<[number, number] | null>(null);
  const [plateWidth, setPlateWidth] = useState(40);
  const [plateDepth, setPlateDepth] = useState(16);
  const [plateRotation, setPlateRotation] = useState(0);
  const [plateRelief, setPlateRelief] = useState(0.6);
  const [trackColor, setTrackColor] = useState("#dc4628");
  const [buildingColor, setBuildingColor] = useState("#b0b0b0");
  const [fmt, setFmt] = useState("3mf");
  const [busy, setBusy] = useState(false);
  // Kept as a key rather than a rendered sentence, so a language switch
  // re-reads the message too (backend error text passes through as-is).
  const [status, setStatus] = useState<Status>(null);
  const [glb, setGlb] = useState<ArrayBuffer | null>(null);
  const [bbox, setBbox] = useState<Bbox | null>(null);
  const [shape, setShape] = useState<Shape>("rect");
  const [rotation, setRotation] = useState(0);
  // Whole track as parsed (`times` = epoch seconds, null when the file has no
  // <time>), plus the time window the user kept.
  const [track, setTrack] = useState<{ pts: [number, number][]; times: number[] | null }>({ pts: [], times: null });
  const [timeSpan, setTimeSpan] = useState<[number, number] | null>(null);
  const [range, setRange] = useState<[number, number] | null>(null);
  const [locked, setLocked] = useState<Lock | null>(null);
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

  useEffect(() => {
    rememberLang(lang);
    document.documentElement.lang = lang;
  }, [lang]);

  // Parse the GPX in the browser: track polyline for the map, plus the
  // automatic extent (track + 8% margin, in the current shape/rotation) as the
  // initial model outline.
  useEffect(() => {
    const clear = () => {
      setBbox(null);
      setTrack({ pts: [], times: null });
      setTimeSpan(null);
      setRange(null);
    };
    if (!file) { clear(); return; }
    let stale = false;
    file.text().then((text) => {
      if (stale) return;
      const doc = new DOMParser().parseFromString(text, "application/xml");
      let els = doc.getElementsByTagNameNS("*", "trkpt");
      if (!els.length) els = doc.getElementsByTagNameNS("*", "rtept");
      if (!els.length) els = doc.getElementsByTagNameNS("*", "wpt");
      const pts: [number, number][] = [];
      const times: number[] = [];
      let last = NaN, t0 = Infinity, t1 = -Infinity;
      for (const p of Array.from(els)) {
        const lon = Number(p.getAttribute("lon"));
        const lat = Number(p.getAttribute("lat"));
        if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
        // Unstamped points inherit the previous stamp (mirrors the backend),
        // so a partially stamped file still trims as one continuous track.
        const ms = Date.parse(p.getElementsByTagNameNS("*", "time")[0]?.textContent ?? "");
        if (Number.isFinite(ms)) {
          last = ms / 1000;
          t0 = Math.min(t0, last);
          t1 = Math.max(t1, last);
        }
        pts.push([lat, lon]);
        times.push(last);
      }
      if (!pts.length) { clear(); return; }
      // A too-short window (or a file stamped with a single instant) is not
      // worth a slider — the whole track is the only sensible selection.
      const span: [number, number] | null = t1 - t0 >= 60 ? [t0, t1] : null;
      setTrack({ pts, times: span ? times.map((t) => (Number.isFinite(t) ? t : t0)) : null });
      setTimeSpan(span);
      setRange(span);
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

  // Trimming is off whenever the window still covers the whole log, so an
  // untouched slider changes nothing that reaches the backend.
  const trimmed = !!(timeSpan && range && (range[0] > timeSpan[0] || range[1] < timeSpan[1]));
  const timeRangeParam = trimmed && range ? `${range[0].toFixed(3)},${range[1].toFixed(3)}` : "";
  // The kept leg: what gets printed, and what 軌跡に合わせる frames.
  const selPts = useMemo(() => {
    const { pts, times } = track;
    if (!trimmed || !times || !range) return pts;
    return pts.filter((_, i) => times[i] >= range[0] && times[i] <= range[1]);
  }, [track, trimmed, range]);
  const mapPts = useMemo(() => thin(track.pts), [track]);
  const mapSel = useMemo(() => (trimmed ? thin(selPts) : null), [trimmed, selPts]);

  // An outing crossing midnight needs the date on every clock reading.
  const multiDay = !!timeSpan &&
    new Date(timeSpan[0] * 1000).toDateString() !== new Date(timeSpan[1] * 1000).toDateString();
  const startDateText = timeSpan
    ? new Date(timeSpan[0] * 1000).toLocaleDateString(localeOf(lang), { year: "numeric", month: "long", day: "numeric" })
    : "";

  const spanM = bbox ? spanMeters(bbox, shape) : null;
  // While the scale is locked it is the anchor; otherwise it's derived.
  const scaleDenom =
    locked === "scale" && lockedDenom ? lockedDenom
    : spanM ? (spanM * 1000) / sizeMm : null;

  // The padlocks are toggles: clicking the closed one releases it, leaving
  // nothing locked.
  function lockTo(item: Lock) {
    if (locked === item) { setLocked(null); return; }
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
    } else {
      // 範囲 locked, or nothing locked: the frame the user drew on the map
      // stays put and the print size takes the change.
      setSizeMm(Math.min(SIZE_MAX, Math.max(SIZE_MIN, (spanMeters(b, shape) * 1000) / d)));
    }
  }

  const extentText = (() => {
    if (!bbox) return "—";
    const [w, h] = extentMeters(bbox);
    const f = (m: number) => (m >= 1000 ? `${(m / 1000).toFixed(2)}km` : `${Math.round(m)}m`);
    return `${f(w)} × ${f(h)}`;
  })();

  // The plate footprint on the map, in real metres. Kept inside the outline
  // here (not only while dragging) so a resized model or a widened plate
  // pulls it back in instead of silently printing a cut-off plaque.
  const mPerMm = spanM ? spanM / sizeMm : null;
  const plateBox = useMemo(() => {
    if (!includePlate || !bbox || !mPerMm || !plateCenter) return null;
    const wM = plateWidth * mPerMm, hM = plateDepth * mPerMm;
    return {
      center: clampPlate(plateCenter, bbox, shape, rotation, wM, hM, plateRotation),
      wM, hM, deg: plateRotation, minM: PLATE_MIN_MM * mPerMm,
    };
  }, [includePlate, bbox, mPerMm, plateCenter, shape, rotation, plateWidth, plateDepth, plateRotation]);

  // Put the plate where the printed track leaves the most room.
  function autoPlacePlate() {
    if (!bbox || !mPerMm) return;
    setPlateCenter(freeSpot(
      selPts, bbox, shape, rotation,
      plateWidth * mPerMm, plateDepth * mPerMm, plateRotation));
  }

  // A plate longer than the model's own long edge could only be printed cut
  // off at both ends, so that is where the corner handles (and the fields)
  // stop.
  const plateMaxMm = Math.min(PLATE_MAX_MM, sizeMm);

  // Map edits come back in metres: the printed size is what the user sets, so
  // the sides are converted back to whole millimetres.
  const onPlateChange = useCallback((e: PlateEdit) => {
    if (e.center) setPlateCenter(e.center);
    if (e.deg !== undefined) setPlateRotation(e.deg);
    if (mPerMm) {
      const mm = (m: number) =>
        Math.min(plateMaxMm, Math.max(PLATE_MIN_MM, Math.round(m / mPerMm)));
      if (e.wM !== undefined) setPlateWidth(mm(e.wM));
      if (e.hM !== undefined) setPlateDepth(mm(e.hM));
    }
  }, [mPerMm, plateMaxMm]);

  const lockBtn = (item: Lock, disabled = false) => (
    <button
      className={`lock-btn${locked === item ? " on" : ""}`}
      title={locked === item ? t.lockOn : t.lockOff}
      disabled={disabled}
      onClick={() => lockTo(item)}
    >
      <LockIcon open={locked !== item} />
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
      f.append("min_color_mm", String(minColor));
      f.append("include_track", String(includeTrack));
      f.append("track_width_mm", String(trackWidth));
      f.append("track_height_mm", String(trackHeight));
      f.append("include_buildings", String(includeBuildings));
      f.append("building_scale", String(buildingScale));
      f.append("min_feature_mm", String(minFeature));
      f.append("terrain_color", TERRAIN_COLOR);
      f.append("track_color", trackColor);
      f.append("building_color", buildingColor);
      if (includePlate && plateSvg && plateBox) {
        f.append("plate_svg", plateSvg);
        f.append("plate_width_mm", String(plateWidth));
        f.append("plate_depth_mm", String(plateDepth));
        f.append("plate_rotation_deg", String(plateRotation));
        f.append("plate_relief_mm", String(plateRelief));
        // The backend takes the position in lon,lat — the map frame can be
        // rotated, so a print-frame offset would not survive the round trip.
        f.append("plate_center", `${plateBox.center[1]},${plateBox.center[0]}`);
      }
      if (bboxParam) f.append("bbox", bboxParam);
      if (timeRangeParam) f.append("time_range", timeRangeParam);
      f.append("shape", shape);
      f.append("rotation_deg", String(rotation));
      f.append("fmt", outFmt);
      return f;
    },
    [file, sizeMm, verticalScale, baseThickness, gridMax, minColor, includeTrack, trackWidth, trackHeight, includeBuildings, buildingScale, minFeature, trackColor, buildingColor, includePlate, plateSvg, plateBox, plateWidth, plateDepth, plateRotation, plateRelief, bboxParam, timeRangeParam, shape, rotation]
  );

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
      setStatus({ kind: "needFile" });
      return;
    }
    setBusy(true);
    setStatus({ kind: "generating" });
    try {
      const resp = await fetch("/api/generate", { method: "POST", body: buildForm("glb") });
      if (!resp.ok) throw new Error(((await resp.json().catch(() => ({}))) as any).detail ?? `HTTP ${resp.status}`);
      setGlb(await resp.arrayBuffer());
      setStatus(null);
    } catch (e) {
      setStatus({ kind: "error", detail: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  async function download() {
    if (!file) {
      setStatus({ kind: "needFile" });
      return;
    }
    setBusy(true);
    setStatus({ kind: "downloading" });
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
      setStatus({ kind: "downloaded" });
    } catch (e) {
      setStatus({ kind: "error", detail: (e as Error).message });
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
        <span className="brand">
          <BrandMark />
          <h1 className="app-title">3D-FOOTPRINT</h1>
        </span>
        <p className="app-subtitle">{t.subtitle}</p>
        <span className="lang-switch" role="group" aria-label="Language">
          {(["ja", "en"] as Lang[]).map((l) => (
            <button
              key={l} className={lang === l ? "on" : ""} aria-pressed={lang === l}
              onClick={() => setLang(l)}
            >
              {l === "ja" ? "日本語" : "English"}
            </button>
          ))}
        </span>
        <span className={`health${health === "ok" ? " ok" : ""}`}>
          <span className="dot" />API {health === "ok" ? t.apiOk : health}
        </span>
      </header>

      <div className="layout">
        <div className="card panel">
          <div className="panel-scroll">
            <label
              className={`dropzone${dragOver ? " drag" : ""}${file ? " has-file" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
            >
              <input type="file" accept=".gpx" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
              {file ? (
                <>
                  <div className="file-name">{file.name}</div>
                  <div className="sub">{t.dropReplace}</div>
                </>
              ) : (
                <>
                  <div>{t.dropGpx}</div>
                  <div className="sub">{t.dropGpxSub}</div>
                </>
              )}
            </label>

            {file && (
              <>
                <h3 className="section-title">{t.secTime}</h3>
                {timeSpan && range ? (
                  <>
                    <div className="row">
                      <label>{startDateText}</label>
                      <span className="val">
                        {clockText(range[0], multiDay)} – {clockText(range[1], multiDay)}
                      </span>
                    </div>
                    <RangeSlider
                      min={timeSpan[0]} max={timeSpan[1]} step={1}
                      value={range} labels={[t.rangeStart, t.rangeEnd]} onChange={setRange}
                    />
                    <div className="range-ends">
                      <span>{clockText(timeSpan[0], multiDay)}</span>
                      <span>
                        {t.selected} {t.duration(range[1] - range[0])}
                        {trimmed && ` / ${t.whole} ${t.duration(timeSpan[1] - timeSpan[0])}`}
                      </span>
                      <span>{clockText(timeSpan[1], multiDay)}</span>
                    </div>
                    <div className="row presets">
                      <button className="btn btn-ghost btn-xs" disabled={!trimmed} onClick={() => setRange(timeSpan)}>
                        {t.resetRange}
                      </button>
                    </div>
                  </>
                ) : (
                  <p className="hint">{t.noTimeHint}</p>
                )}
              </>
            )}

            <h3 className="section-title">{t.secSize}</h3>
            <div className="row">
              {lockBtn("span", !bbox)}
              <label>{t.extent}</label>
              <span className="val">{extentText}</span>
            </div>
            <div className="row">
              {lockBtn("size")}
              <label>{t.printSize}</label>
              <NumField value={sizeMm} min={SIZE_MIN} max={SIZE_MAX} step={5} digits={1} disabled={locked === "size"} onCommit={commitSize} />
            </div>
            <div className="row">
              {lockBtn("scale", !bbox)}
              <label>{t.scale}</label>
              <NumField value={scaleDenom == null ? null : Math.round(scaleDenom)} min={100} max={10000000} step={1000} disabled={!bbox || locked === "scale"} onCommit={commitScale} />
            </div>
            {scaleDenom != null && (
              <div className="row presets">
                {niceScales(scaleDenom).map((d) => (
                  <button key={d} className="btn btn-ghost btn-xs scale-preset" disabled={locked === "scale"} onClick={() => commitScale(d)}>
                    1:{d.toLocaleString()}
                  </button>
                ))}
              </div>
            )}

            <h3 className="section-title">{t.secModel}</h3>
            <div className="row">
              <label>{t.verticalScale}<span className="val">×{verticalScale}</span></label>
              <input type="range" min={1} max={30} step={0.5} value={verticalScale} onChange={(e) => setVerticalScale(Number(e.target.value))} />
            </div>
            <div className="row">
              <label>{t.baseThickness}</label>
              <input type="number" min={0} max={20} step={0.5} value={baseThickness} onChange={(e) => setBaseThickness(Number(e.target.value))} />
            </div>
            <div className="row">
              <label>{t.resolution}</label>
              <select value={gridMax} onChange={(e) => setGridMax(Number(e.target.value))}>
                <option value={700}>{t.resStandard}</option>
                <option value={1000}>{t.resHigh}</option>
                <option value={1400}>{t.resMax}</option>
              </select>
            </div>

            <h3 className="section-title">{t.secColor}</h3>
            <div className="row">
              <label>{t.minColor}<span className="val">{minColor ? `${minColor}mm` : t.minColorOff}</span></label>
              <input type="range" min={0} max={4} step={0.5} value={minColor} onChange={(e) => setMinColor(Number(e.target.value))} />
            </div>
            <p className="hint">{t.minColorHint}</p>

            <h3 className="section-title">{t.secTrack}</h3>
            <div className="row">
              <label>{t.includeTrack}</label>
              <input className="toggle" type="checkbox" checked={includeTrack} onChange={(e) => setIncludeTrack(e.target.checked)} />
            </div>
            <div className={`row${includeTrack ? "" : " dim"}`}>
              <label>{t.trackWidth}</label>
              <input type="number" min={0.4} max={10} step={0.1} value={trackWidth} disabled={!includeTrack} onChange={(e) => setTrackWidth(Number(e.target.value))} />
            </div>
            <div className={`row${includeTrack ? "" : " dim"}`}>
              <label>{t.trackHeight}</label>
              <input type="number" min={0.2} max={10} step={0.1} value={trackHeight} disabled={!includeTrack} onChange={(e) => setTrackHeight(Number(e.target.value))} />
            </div>
            <div className={`row${includeTrack ? "" : " dim"}`}>
              <label>{t.trackColor}</label>
              <input type="color" value={trackColor} disabled={!includeTrack} onChange={(e) => setTrackColor(e.target.value)} />
            </div>

            <h3 className="section-title">{t.secBuildings}</h3>
            <div className="row">
              <label>{t.includeBuildings}</label>
              <input className="toggle" type="checkbox" checked={includeBuildings} onChange={(e) => setIncludeBuildings(e.target.checked)} />
            </div>
            {includeBuildings && (
              <>
                <p className="hint">{t.buildingsHint}</p>
                <div className="row">
                  <label>{t.buildingScale}<span className="val">×{buildingScale}</span></label>
                  <input type="range" min={1} max={50} step={1} value={buildingScale} onChange={(e) => setBuildingScale(Number(e.target.value))} />
                </div>
                <div className="row">
                  <label>{t.minFeature}<span className="val">{minFeature}mm</span></label>
                  <input type="range" min={0.4} max={2} step={0.1} value={minFeature} onChange={(e) => setMinFeature(Number(e.target.value))} />
                </div>
                <div className="row">
                  <label>{t.buildingColor}</label>
                  <input type="color" value={buildingColor} onChange={(e) => setBuildingColor(e.target.value)} />
                </div>
              </>
            )}

            <h3 className="section-title">{t.secPlate}</h3>
            <div className="row">
              <label>{t.includePlate}</label>
              <input
                className="toggle" type="checkbox" checked={includePlate}
                onChange={(e) => {
                  setIncludePlate(e.target.checked);
                  if (e.target.checked && !plateCenter) autoPlacePlate();
                }}
              />
            </div>
            {includePlate && (
              <>
                <label className={`dropzone slim${plateSvg ? " has-file" : ""}`}>
                  <input type="file" accept=".svg,image/svg+xml" onChange={(e) => setPlateSvg(e.target.files?.[0] ?? null)} />
                  {plateSvg ? (
                    <div className="file-name">{plateSvg.name}</div>
                  ) : (
                    <div>{t.plateSvg}</div>
                  )}
                  <div className="sub">{t.plateFit(plateWidth, plateDepth)}</div>
                </label>
                {plateUrl && (
                  <div className="plate-preview" style={{ aspectRatio: `${plateWidth} / ${plateDepth}`, background: TERRAIN_COLOR }}>
                    <img src={plateUrl} alt={t.platePreview} />
                  </div>
                )}
                <p className="hint">{rich(t.plateHint)}</p>
                <div className="row presets">
                  <button className="btn btn-ghost btn-xs" disabled={!bbox} onClick={autoPlacePlate}>
                    {t.platePlace}
                  </button>
                  <span className="val">{t.plateHandles}</span>
                </div>
                <div className="row">
                  <label>{t.plateWidth}</label>
                  <input type="number" min={PLATE_MIN_MM} max={plateMaxMm} step={1} value={plateWidth} onChange={(e) => setPlateWidth(Number(e.target.value))} />
                </div>
                <div className="row">
                  <label>{t.plateDepth}</label>
                  <input type="number" min={PLATE_MIN_MM} max={plateMaxMm} step={1} value={plateDepth} onChange={(e) => setPlateDepth(Number(e.target.value))} />
                </div>
                <div className="row">
                  <label>{t.plateRotation}</label>
                  <input type="number" step={5} value={plateRotation} onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) setPlateRotation(v); }} />
                </div>
                <div className="row">
                  <label>{t.plateRelief}</label>
                  <input type="number" min={0.2} max={3} step={0.1} value={plateRelief} onChange={(e) => setPlateRelief(Number(e.target.value))} />
                </div>
              </>
            )}

          </div>

          <div className="panel-actions">
            <button className="btn btn-primary btn-block" onClick={createModel} disabled={busy || !file}>
              {busy && <span className="spinner" />}
              {t.create}
            </button>

            <div className="row" style={{ marginTop: 6 }}>
              <label>{t.format}</label>
              <select value={fmt} onChange={(e) => setFmt(e.target.value)}>
                <option value="3mf">{t.fmt3mf}</option>
                <option value="stl_multi">{t.fmtStlMulti}</option>
                <option value="stl">{t.fmtStl}</option>
              </select>
            </div>
            <button className="btn btn-secondary btn-block" onClick={download} disabled={busy || !file}>
              {t.download}
            </button>
            {status && (
              <p className={`status${status.kind === "error" ? " error" : ""}`}>
                {status.kind === "error"
                  ? `${t.errorPrefix}${status.detail}`
                  : t[STATUS_TEXT[status.kind]]}
              </p>
            )}
          </div>
        </div>

        <div className="stack">
          <div className="card map-card">
            <div className="card-head">
              <strong>{t.mapTitle}</strong>
              <select value={shape} onChange={(e) => onShapeChange(e.target.value as Shape)}>
                <option value="rect">{t.shapeRect}</option>
                <option value="square">{t.shapeSquare}</option>
                <option value="hex">{t.shapeHex}</option>
              </select>
              <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                {t.rotation}
                <input
                  type="number" step={5} value={rotation} style={{ width: 62 }}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (Number.isFinite(v)) setRotation(v);
                  }}
                />°
              </label>
              <span>{locked === "span" ? t.handlesLocked : t.handles}</span>
              {bbox && (
                <span className="coords">
                  {t.center} {(((bbox[1] + bbox[3]) / 2)).toFixed(4)}, {(((bbox[0] + bbox[2]) / 2)).toFixed(4)}
                </span>
              )}
              <span style={{ flex: 1 }} />
              <button
                className="btn btn-ghost"
                onClick={() => {
                  const bb = fitBbox(selPts, shape, rotation);
                  if (!bb) return;
                  if (locked === "span" && bbox) recentre(bb); else applyBbox(bb);
                }}
                disabled={!selPts.length}
              >
                {locked === "span" ? t.centerTrack : t.fitTrack}
              </button>
            </div>
            <div className="card-body map-box">
              {!file && <div className="overlay-hint"><span>{t.mapEmpty}</span></div>}
              <MapPicker
                points={mapPts} selection={mapSel} bbox={bbox} shape={shape} rotation={rotation}
                resizable={locked !== "span"} plate={plateBox}
                onBboxChange={onBboxChange} onRotationChange={setRotation}
                onPlateChange={onPlateChange}
              />
            </div>
          </div>

          <div className="card preview-card">
            <div className="card-head">
              <strong>{t.previewTitle}</strong>
              <span>{t.previewHelp}</span>
            </div>
            <div className="card-body preview-box">
              {!glb && <div className="overlay-hint"><span>{t.previewEmpty}</span></div>}
              {busy && <div className="busy-badge"><span className="spinner" />{t.busy}</div>}
              {glb && (
                <div className="preview-legend">
                  {legendOf(t).map(([name, c]) => (
                    <span key={name}><i style={{ background: c }} />{name}</span>
                  ))}
                </div>
              )}
              <Preview glb={glb} />
            </div>
          </div>
        </div>
      </div>

      <footer className="credits">
        {t.credits}{" "}
        <a href="https://maps.gsi.go.jp/development/ichiran.html" target="_blank" rel="noreferrer">{t.gsi}</a>
        {t.creditSep}
        <a href="https://www.mlit.go.jp/plateau/" target="_blank" rel="noreferrer">{t.plateau}</a>
        {t.creditSep}
        <a href="https://www.eorc.jaxa.jp/ALOS/jp/dataset/lulc_j.htm" target="_blank" rel="noreferrer">{t.jaxa}</a>
        {t.creditSep}
        {t.osm} <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a>{t.osmSuffix}
      </footer>
    </main>
  );
}

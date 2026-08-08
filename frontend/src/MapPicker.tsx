import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

export type Bbox = [number, number, number, number]; // west, south, east, north
export type Shape = "rect" | "square" | "hex";

// Equirectangular metres-per-degree — must match backend mesh.py so a
// "square"/"regular hexagon" here is square/regular in printed millimetres.
export const M_PER_DEG_LAT = 110540;
export const M_PER_DEG_LON = 111320;
const RAD = Math.PI / 180;
const SQ3 = Math.sqrt(3);
const MIN_SIDE_M = 130; // smallest draggable side (> backend's 0.001 deg floor)

// Enforce the backend's minimum span (0.001 deg per side) and the shape's
// aspect ratio (square 1:1, flat-top hexagon 2R x sqrt(3)R in metres),
// expanding around the centre. Stable under re-application.
export function normalizeBbox(bb: Bbox, shape: Shape): Bbox {
  let [w, s, e, n] = bb;
  if (e - w < 1e-3) { const c = (w + e) / 2; w = c - 5e-4; e = c + 5e-4; }
  if (n - s < 1e-3) { const c = (s + n) / 2; s = c - 5e-4; n = c + 5e-4; }
  if (shape === "rect") return [w, s, e, n];
  const clat = (s + n) / 2, clon = (w + e) / 2;
  const mlon = M_PER_DEG_LON * Math.cos(clat * RAD);
  let hw = ((e - w) / 2) * mlon;
  let hh = ((n - s) / 2) * M_PER_DEG_LAT;
  if (shape === "square") {
    hw = hh = Math.max(hw, hh);
  } else {
    const r = Math.max(hw, (2 * hh) / SQ3);
    hw = r; hh = (SQ3 / 2) * r;
  }
  return [clon - hw / mlon, clat - hh / M_PER_DEG_LAT,
          clon + hw / mlon, clat + hh / M_PER_DEG_LAT];
}

// Smallest bbox of `shape` at `rotationDeg` whose outline encloses the whole
// track, plus an 8% margin (the automatic extent, like the backend's).
export function fitBbox(
  pts: [number, number][], shape: Shape, rotationDeg: number
): Bbox | null {
  if (!pts.length) return null;
  const clat0 = pts.reduce((a, p) => a + p[0], 0) / pts.length;
  const clon0 = pts.reduce((a, p) => a + p[1], 0) / pts.length;
  const mlon = M_PER_DEG_LON * Math.cos(clat0 * RAD);
  const th = rotationDeg * RAD, c = Math.cos(th), sn = Math.sin(th);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [lat, lon] of pts) {
    const x = (lon - clon0) * mlon, y = (lat - clat0) * M_PER_DEG_LAT;
    const xl = x * c + y * sn, yl = -x * sn + y * c; // into the shape's frame
    x0 = Math.min(x0, xl); x1 = Math.max(x1, xl);
    y0 = Math.min(y0, yl); y1 = Math.max(y1, yl);
  }
  const mx = Math.max(x1 - x0, 100) * 0.08, my = Math.max(y1 - y0, 100) * 0.08;
  x0 -= mx; x1 += mx; y0 -= my; y1 += my;
  let hw = (x1 - x0) / 2, hh = (y1 - y0) / 2;
  if (shape === "square") hw = hh = Math.max(hw, hh);
  if (shape === "hex") {
    // Smallest regular hexagon containing the fitted rect (corners included).
    const r = Math.max(hw + hh / SQ3, (2 * hh) / SQ3);
    hw = r; hh = (SQ3 / 2) * r;
  }
  const lx = (x0 + x1) / 2, ly = (y0 + y1) / 2;      // rect centre, shape frame
  const cx = lx * c - ly * sn, cy = lx * sn + ly * c; // back to world
  const clon = clon0 + cx / mlon, clat = clat0 + cy / M_PER_DEG_LAT;
  return normalizeBbox(
    [clon - hw / mlon, clat - hh / M_PER_DEG_LAT,
     clon + hw / mlon, clat + hh / M_PER_DEG_LAT], shape);
}

// ---- size / scale helpers (must mirror backend region.half_extents_m) ------

// Long edge of the model outline in metres — what size_mm maps onto
// (backend region.span_m): rect = bbox long edge, square/hex = inscribed.
export function spanMeters(bb: Bbox, shape: Shape): number {
  const g = geoOf(bb);
  if (shape === "square") return 2 * Math.min(g.hw, g.hh);
  if (shape === "hex") return 2 * Math.min(g.hw, (2 * g.hh) / SQ3);
  return 2 * Math.max(g.hw, g.hh);
}

// Printed footprint extents (width, height) in metres.
export function extentMeters(bb: Bbox): [number, number] {
  const g = geoOf(bb);
  return [2 * g.hw, 2 * g.hh];
}

// Uniformly rescale the bbox about its centre by factor k (aspect preserved,
// so the shape stays a true square / regular hexagon).
export function scaleBbox(bb: Bbox, k: number): Bbox {
  const clon = (bb[0] + bb[2]) / 2, clat = (bb[1] + bb[3]) / 2;
  const hw = ((bb[2] - bb[0]) / 2) * k, hh = ((bb[3] - bb[1]) / 2) * k;
  return [clon - hw, clat - hh, clon + hw, clat + hh];
}

// ---- local geometry helpers -------------------------------------------------

type Geo = { clat: number; clon: number; mlon: number; hw: number; hh: number };

function geoOf(bb: Bbox): Geo {
  const clat = (bb[1] + bb[3]) / 2, clon = (bb[0] + bb[2]) / 2;
  const mlon = M_PER_DEG_LON * Math.cos(clat * RAD);
  return {
    clat, clon, mlon,
    hw: ((bb[2] - bb[0]) / 2) * mlon,
    hh: ((bb[3] - bb[1]) / 2) * M_PER_DEG_LAT,
  };
}

// Local metres (rotated by th about the centre) -> [lat, lng].
function ll(g: Geo, x: number, y: number, th: number): L.LatLngTuple {
  const xr = x * Math.cos(th) - y * Math.sin(th);
  const yr = x * Math.sin(th) + y * Math.cos(th);
  return [g.clat + yr / M_PER_DEG_LAT, g.clon + xr / g.mlon];
}

// Inverse of ll: [lat, lng] -> the shape's local metre frame.
function local(g: Geo, lat: number, lon: number, th: number): [number, number] {
  const x = (lon - g.clon) * g.mlon, y = (lat - g.clat) * M_PER_DEG_LAT;
  return [x * Math.cos(th) + y * Math.sin(th), -x * Math.sin(th) + y * Math.cos(th)];
}

// ---- nameplate placement inside the outline ---------------------------------

// The nameplate's footprint on the map. `deg` is measured against the model
// outline, not north, so turning the model carries the plate around with it;
// `minM` is the smallest printable side in metres (the caller owns the mm).
export type Plate = {
  center: [number, number];
  wM: number;
  hM: number;
  deg: number;
  minM: number;
};

// The outline as half-planes |n·p| <= d in the local frame (both shapes are
// convex and centre-symmetric): rect/square two, flat-top hexagon three.
function halfPlanes(bb: Bbox, shape: Shape): [number, number, number][] {
  const g = geoOf(bb);
  if (shape === "hex") {
    const d = (SQ3 / 2) * Math.min(g.hw, (2 * g.hh) / SQ3);
    return [30, 90, 150].map((a) => [Math.cos(a * RAD), Math.sin(a * RAD), d]);
  }
  const [hw, hh] = shape === "square"
    ? [Math.min(g.hw, g.hh), Math.min(g.hw, g.hh)] : [g.hw, g.hh];
  return [[1, 0, hw], [0, 1, hh]];
}

// Pull a plate centre back until the whole plate sits inside the outline.
// Each half-plane is shrunk by the plate's own reach along that normal (the
// turned box's support function), so the rectangle — not just its centre —
// is what stays in. Corners need the passes: fixing one plane can push the
// centre past another.
export function clampPlate(
  center: [number, number], bb: Bbox, shape: Shape, rotationDeg: number,
  wM: number, hM: number, plateDeg: number,
): [number, number] {
  const g = geoOf(bb), th = rotationDeg * RAD, ph = plateDeg * RAD;
  const [ux, uy] = [Math.cos(ph), Math.sin(ph)];   // the plate's own axes
  let [x, y] = local(g, center[0], center[1], th);
  for (let pass = 0; pass < 3; pass++) {
    for (const [nx, ny, d] of halfPlanes(bb, shape)) {
      const reach = (wM / 2) * Math.abs(nx * ux + ny * uy)
                  + (hM / 2) * Math.abs(-nx * uy + ny * ux);
      const lim = d - reach;
      const t = x * nx + y * ny;
      // lim < 0: the plate is wider than the model here — centre it and let
      // the backend cut the overhang flush with the model edge.
      const push = lim < 0 ? t : Math.sign(t) * Math.max(Math.abs(t) - lim, 0);
      x -= push * nx;
      y -= push * ny;
    }
  }
  return ll(g, x, y, th) as [number, number];
}

// The spot inside the outline whose plate clears the track by the most — what
// 「軌跡を避けて配置」 picks. Candidates are a grid over the outline, each
// pulled inside first; clearance is the distance from the plate rectangle to
// the nearest track point, measured in the plate's own frame so its angle
// counts. With no track the model centre wins (nothing beats infinity).
export function freeSpot(
  pts: [number, number][], bb: Bbox, shape: Shape, rotationDeg: number,
  wM: number, hM: number, plateDeg: number,
): [number, number] {
  const g = geoOf(bb), th = rotationDeg * RAD, ph = plateDeg * RAD;
  const c = Math.cos(ph), s = Math.sin(ph);
  const stride = Math.max(1, Math.ceil(pts.length / 800));
  const track = pts.filter((_, i) => i % stride === 0)
    .map(([lat, lon]) => local(g, lat, lon, th));
  const clearance = (x: number, y: number) => {
    let best = Infinity;
    for (const [px, py] of track) {
      const ax = px - x, ay = py - y;
      const dx = Math.max(Math.abs(ax * c + ay * s) - wM / 2, 0);
      const dy = Math.max(Math.abs(-ax * s + ay * c) - hM / 2, 0);
      const d = dx * dx + dy * dy;
      if (d < best) { best = d; if (d === 0) break; }
    }
    return best;
  };
  const fit = (p: [number, number]) =>
    clampPlate(p, bb, shape, rotationDeg, wM, hM, plateDeg);
  const N = 25;
  let best = fit([g.clat, g.clon]);
  let bestScore = clearance(...local(g, best[0], best[1], th));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const cand = fit(ll(g, g.hw * (2 * i / (N - 1) - 1),
                          g.hh * (2 * j / (N - 1) - 1), th) as [number, number]);
      const score = clearance(...local(g, cand[0], cand[1], th));
      if (score > bestScore) { bestScore = score; best = cand; }
    }
  }
  return best;
}

// The plate's own frame (metres about its centre, turned by the model's
// rotation and its own) <-> [lat, lng].
function plateFrame(p: Plate, rotationDeg: number) {
  const th = (rotationDeg + p.deg) * RAD;
  const c = Math.cos(th), s = Math.sin(th);
  const mlon = M_PER_DEG_LON * Math.cos(p.center[0] * RAD);
  return {
    at: (x: number, y: number): L.LatLngTuple => [
      p.center[0] + (x * s + y * c) / M_PER_DEG_LAT,
      p.center[1] + (x * c - y * s) / mlon,
    ],
    of: (lat: number, lng: number): [number, number] => {
      const dx = (lng - p.center[1]) * mlon, dy = (lat - p.center[0]) * M_PER_DEG_LAT;
      return [dx * c + dy * s, -dx * s + dy * c];
    },
  };
}

function outlinePts(g: Geo, shape: Shape, th: number): L.LatLngTuple[] {
  if (shape === "hex") { // flat-top regular hexagon, circumradius = hw
    return [0, 1, 2, 3, 4, 5].map((k) =>
      ll(g, g.hw * Math.cos(k * 60 * RAD), g.hw * Math.sin(k * 60 * RAD), th));
  }
  return [ll(g, -g.hw, -g.hh, th), ll(g, g.hw, -g.hh, th),
          ll(g, g.hw, g.hh, th), ll(g, -g.hw, g.hh, th)];
}

function framePts(g: Geo, th: number): L.LatLngTuple[] {
  return [ll(g, -g.hw, -g.hh, th), ll(g, g.hw, -g.hh, th),
          ll(g, g.hw, g.hh, th), ll(g, -g.hw, g.hh, th)];
}

function rotHandleXY(g: Geo): [number, number] {
  return [0, g.hh + Math.max(0.22 * Math.max(g.hw, g.hh), 40)];
}

const TRACK_STYLE: L.PolylineOptions = { color: "#dc4628", weight: 3, opacity: 1 };
const OUT_OF_RANGE_STYLE: L.PolylineOptions = { color: "#7d8794", weight: 2, opacity: 0.5 };

// Handle icons as plain divs (Leaflet's default marker PNGs don't survive
// bundling, and we want dedicated resize/rotate/move affordances anyway).
const cornerIcon = L.divIcon({
  className: "",
  html: '<div style="width:14px;height:14px;background:#fff;border:2px solid #d33;border-radius:3px;box-shadow:0 1px 3px #0006;cursor:nwse-resize"></div>',
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});
const rotateIcon = L.divIcon({
  className: "",
  html: '<div style="width:16px;height:16px;background:#fff;border:2px solid #d33;border-radius:50%;box-shadow:0 1px 3px #0006;cursor:grab"></div>',
  iconSize: [16, 16],
  iconAnchor: [8, 8],
});
const moveIcon = L.divIcon({
  className: "",
  html: '<div style="width:22px;height:22px;display:flex;align-items:center;justify-content:center;background:#fff;border:2px solid #d33;border-radius:50%;box-shadow:0 1px 3px #0006;cursor:move;font-size:13px;line-height:1;color:#d33">✥</div>',
  iconSize: [22, 22],
  iconAnchor: [11, 11],
});
// The nameplate carries the same three affordances in its own colour.
const PLATE_COLOR = "#2e5e8c";
const plateIcon = L.divIcon({
  className: "",
  html: `<div style="width:20px;height:20px;display:flex;align-items:center;justify-content:center;background:#fff;border:2px solid ${PLATE_COLOR};border-radius:50%;box-shadow:0 1px 3px #0006;cursor:move;font-size:12px;line-height:1;color:${PLATE_COLOR}">✥</div>`,
  iconSize: [20, 20],
  iconAnchor: [10, 10],
});
const plateCornerIcon = L.divIcon({
  className: "",
  html: `<div style="width:12px;height:12px;background:#fff;border:2px solid ${PLATE_COLOR};border-radius:2px;box-shadow:0 1px 3px #0006;cursor:nwse-resize"></div>`,
  iconSize: [12, 12],
  iconAnchor: [6, 6],
});
const plateRotateIcon = L.divIcon({
  className: "",
  html: `<div style="width:14px;height:14px;background:#fff;border:2px solid ${PLATE_COLOR};border-radius:50%;box-shadow:0 1px 3px #0006;cursor:grab"></div>`,
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

export type PlateEdit = {
  center?: [number, number];
  wM?: number;
  hM?: number;
  deg?: number;
};

// OSM slippy map showing the GPX track, with the model outline (rect / square /
// regular hexagon, rotatable) the user edits with three handles: corner ■ =
// resize, ● above the top edge = rotate, centre ✥ = move. The nameplate
// footprint carries the same three handles in blue — the track is right there
// to place it clear of.
export function MapPicker({ points, selection = null, bbox, shape, rotation, resizable = true, plate = null, onBboxChange, onRotationChange, onPlateChange }: {
  points: [number, number][]; // [lat, lon] in track order
  selection?: [number, number][] | null; // in-time-range part, null = all of it
  bbox: Bbox | null;          // unrotated shape extents
  shape: Shape;
  rotation: number;           // deg CCW
  resizable?: boolean;        // false: fixed-extent frame, pan/rotate only
  plate?: Plate | null;       // nameplate footprint (metres), null = no plate
  onBboxChange: (bb: Bbox) => void;
  onRotationChange: (deg: number) => void;
  onPlateChange?: (edit: PlateEdit) => void;
}) {
  const divRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map>();
  const lineRef = useRef<L.Polyline>();
  const selLineRef = useRef<L.Polyline>();
  const outlineRef = useRef<L.Polygon>();
  const frameRef = useRef<L.Polygon>();   // dashed bbox, shown for hex
  const tetherRef = useRef<L.Polyline>(); // top edge -> rotate handle
  const swRef = useRef<L.Marker>();
  const neRef = useRef<L.Marker>();
  const mvRef = useRef<L.Marker>();
  const rotMarkRef = useRef<L.Marker>();
  const plateBoxRef = useRef<L.Polygon>();
  const plateTetherRef = useRef<L.Polyline>();
  const plateMkRef = useRef<L.Marker>();
  const plateSwRef = useRef<L.Marker>();
  const plateNeRef = useRef<L.Marker>();
  const plateRotRef = useRef<L.Marker>();

  // Latest values for use inside Leaflet event handlers.
  const bboxRef = useRef(bbox); bboxRef.current = bbox;
  const shapeRef = useRef(shape); shapeRef.current = shape;
  const rotRef = useRef(rotation); rotRef.current = rotation;
  const plateRef = useRef(plate); plateRef.current = plate;
  const cbBoxRef = useRef(onBboxChange); cbBoxRef.current = onBboxChange;
  const cbRotRef = useRef(onRotationChange); cbRotRef.current = onRotationChange;
  const cbPlateRef = useRef(onPlateChange); cbPlateRef.current = onPlateChange;

  useEffect(() => {
    const map = L.map(divRef.current!, { zoomSnap: 0.5 });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
    map.setView([35.68, 139.76], 10); // placeholder until the track arrives
    mapRef.current = map;
    const ro = new ResizeObserver(() => map.invalidateSize());
    ro.observe(divRef.current!);
    return () => {
      ro.disconnect();
      map.remove();
    };
  }, []);

  // Track polyline; recenter whenever a new file is parsed.
  useEffect(() => {
    const map = mapRef.current!;
    lineRef.current?.remove();
    lineRef.current = undefined;
    if (!points.length) return;
    lineRef.current = L.polyline(points, TRACK_STYLE).addTo(map);
    map.fitBounds(lineRef.current.getBounds().pad(0.15));
  }, [points]);

  // Time-trimmed track: the selected leg keeps the track colour and the rest
  // stays as a faint reminder of where it came from. (points dep: the base
  // line is recreated above, so the style and the stacking order go with it.)
  useEffect(() => {
    const map = mapRef.current!;
    selLineRef.current?.remove();
    selLineRef.current = undefined;
    lineRef.current?.setStyle(selection ? OUT_OF_RANGE_STYLE : TRACK_STYLE);
    if (!selection?.length) return;
    selLineRef.current = L.polyline(selection, TRACK_STYLE).addTo(map);
  }, [points, selection]);

  // Reposition every layer from (bb, shape, th). `skip` keeps the marker being
  // dragged where Leaflet has it, so we never fight an active drag.
  function redraw(bb: Bbox, shp: Shape, deg: number, skip?: L.Marker) {
    const map = mapRef.current!;
    const g = geoOf(bb);
    const th = deg * RAD;
    outlineRef.current!.setLatLngs(outlinePts(g, shp, th));
    if (shp === "hex") {
      frameRef.current!.setLatLngs(framePts(g, th));
      if (!map.hasLayer(frameRef.current!)) frameRef.current!.addTo(map);
    } else if (map.hasLayer(frameRef.current!)) {
      frameRef.current!.remove();
    }
    const [rx, ry] = rotHandleXY(g);
    tetherRef.current!.setLatLngs([ll(g, 0, g.hh, th), ll(g, rx, ry, th)]);
    const place = (m: L.Marker | undefined, pos: L.LatLngTuple) => {
      if (m && m !== skip) m.setLatLng(pos);
    };
    place(swRef.current, ll(g, -g.hw, -g.hh, th));
    place(neRef.current, ll(g, g.hw, g.hh, th));
    place(mvRef.current, [g.clat, g.clon]);
    place(rotMarkRef.current, ll(g, rx, ry, th));
  }

  // Drag geometry. Resizing keeps the opposite corner fixed and re-derives the
  // extents in the rotated frame (ratio-locked for square/hex); moving drags
  // the centre; rotating reads the handle's bearing about the centre. During a
  // drag we only redraw; the new bbox/rotation is emitted on dragend and comes
  // back through props (no update loop).
  function resizeTo(dragged: L.Marker, other: L.Marker): Bbox {
    const g = geoOf(bboxRef.current!);
    const th = rotRef.current * RAD, c = Math.cos(th), sn = Math.sin(th);
    const P = dragged.getLatLng(), F = other.getLatLng();
    const fx = (F.lng - g.clon) * g.mlon, fy = (F.lat - g.clat) * M_PER_DEG_LAT;
    const px = (P.lng - g.clon) * g.mlon, py = (P.lat - g.clat) * M_PER_DEG_LAT;
    const dx = (px - fx) * c + (py - fy) * sn;  // diagonal in the shape frame
    const dy = -(px - fx) * sn + (py - fy) * c;
    let w = Math.max(Math.abs(dx), MIN_SIDE_M), h = Math.max(Math.abs(dy), MIN_SIDE_M);
    const shp = shapeRef.current;
    if (shp === "square") w = h = Math.max(w, h);
    if (shp === "hex") { const r = Math.max(w / 2, h / SQ3); w = 2 * r; h = SQ3 * r; }
    const lx = (dx >= 0 ? 1 : -1) * w / 2, lyy = (dy >= 0 ? 1 : -1) * h / 2;
    const cx = fx + lx * c - lyy * sn, cy = fy + lx * sn + lyy * c;
    const clon = g.clon + cx / g.mlon, clat = g.clat + cy / M_PER_DEG_LAT;
    return [clon - w / 2 / g.mlon, clat - h / 2 / M_PER_DEG_LAT,
            clon + w / 2 / g.mlon, clat + h / 2 / M_PER_DEG_LAT];
  }

  function moveTo(): Bbox {
    const b = bboxRef.current!;
    const P = mvRef.current!.getLatLng();
    const dlon = P.lng - (b[0] + b[2]) / 2, dlat = P.lat - (b[1] + b[3]) / 2;
    return [b[0] + dlon, b[1] + dlat, b[2] + dlon, b[3] + dlat];
  }

  function rotateTo(): number {
    const g = geoOf(bboxRef.current!);
    const P = rotMarkRef.current!.getLatLng();
    const vx = (P.lng - g.clon) * g.mlon, vy = (P.lat - g.clat) * M_PER_DEG_LAT;
    let deg = Math.round(Math.atan2(vy, vx) / RAD - 90);
    if (deg <= -180) deg += 360;
    if (deg > 180) deg -= 360;
    return deg;
  }

  // Outline + handles, kept in sync with the region props.
  useEffect(() => {
    const map = mapRef.current!;
    if (!bbox) {
      for (const ref of [outlineRef, frameRef, tetherRef] as const) {
        ref.current?.remove();
        ref.current = undefined;
      }
      for (const ref of [swRef, neRef, mvRef, rotMarkRef] as const) {
        ref.current?.remove();
        ref.current = undefined;
      }
      return;
    }
    if (!outlineRef.current) {
      outlineRef.current = L.polygon([], {
        color: "#d33", weight: 2, fillOpacity: 0.06, interactive: false,
      }).addTo(map);
      frameRef.current = L.polygon([], {
        color: "#d33", weight: 1, dashArray: "4 4", fill: false, interactive: false,
      });
      tetherRef.current = L.polyline([], {
        color: "#d33", weight: 1, dashArray: "2 4", interactive: false,
      }).addTo(map);
      const mk = (icon: L.DivIcon) =>
        L.marker([0, 0], { icon, draggable: true, autoPan: true }).addTo(map);
      swRef.current = mk(cornerIcon);
      neRef.current = mk(cornerIcon);
      mvRef.current = mk(moveIcon);
      rotMarkRef.current = mk(rotateIcon);

      const wire = (m: L.Marker, calc: () => { bb?: Bbox; deg?: number }) => {
        m.on("drag", () => {
          const r = calc();
          redraw(r.bb ?? bboxRef.current!, shapeRef.current,
                 r.deg ?? rotRef.current, m);
        });
        m.on("dragend", () => {
          const r = calc();
          // Snap all handles to the final geometry even if the emitted value
          // normalizes to the same state (no re-render in that case).
          redraw(r.bb ?? bboxRef.current!, shapeRef.current, r.deg ?? rotRef.current);
          if (r.bb) cbBoxRef.current(r.bb);
          if (r.deg !== undefined) cbRotRef.current(r.deg);
        });
      };
      wire(swRef.current, () => ({ bb: resizeTo(swRef.current!, neRef.current!) }));
      wire(neRef.current, () => ({ bb: resizeTo(neRef.current!, swRef.current!) }));
      wire(mvRef.current, () => ({ bb: moveTo() }));
      wire(rotMarkRef.current, () => ({ deg: rotateTo() }));
    }
    redraw(bbox, shape, rotation);
  }, [bbox, shape, rotation]);

  // Nameplate footprint with its own move / resize / rotate handles. Every
  // edit is clamped to the model outline here as well as in the caller, so
  // the rectangle on screen is always the plate that will be printed.
  function redrawPlate(p: Plate, skip?: L.Marker) {
    const { at } = plateFrame(p, rotRef.current);
    const hw = p.wM / 2, hh = p.hM / 2;
    plateBoxRef.current!.setLatLngs([at(-hw, -hh), at(hw, -hh), at(hw, hh), at(-hw, hh)]);
    const reach = hh + Math.max(0.35 * Math.max(hw, hh), 25);
    plateTetherRef.current!.setLatLngs([at(0, hh), at(0, reach)]);
    const place = (m: L.Marker | undefined, pos: L.LatLngTuple) => {
      if (m && m !== skip) m.setLatLng(pos);
    };
    place(plateMkRef.current, p.center);
    place(plateSwRef.current, at(-hw, -hh));
    place(plateNeRef.current, at(hw, hh));
    place(plateRotRef.current, at(0, reach));
  }

  // Every handle answers the same question — what plate does this drag mean —
  // and the answer is clamped back inside the outline before it is drawn or
  // emitted.
  function plateFrom(edit: PlateEdit): Plate {
    const p = plateRef.current!;
    const next = { ...p, ...edit };
    return {
      ...next,
      center: clampPlate(next.center, bboxRef.current!, shapeRef.current,
                         rotRef.current, next.wM, next.hM, next.deg),
    };
  }

  // Resize: the opposite corner stays put and the diagonal, read in the
  // plate's own frame, gives the new sides and centre.
  function plateResize(dragged: L.Marker, other: L.Marker): PlateEdit {
    const p = plateRef.current!;
    const f = plateFrame(p, rotRef.current);
    const D = dragged.getLatLng(), F = other.getLatLng();
    const [dx, dy] = f.of(D.lat, D.lng);
    const [fx, fy] = f.of(F.lat, F.lng);
    const wM = Math.max(Math.abs(dx - fx), p.minM);
    const hM = Math.max(Math.abs(dy - fy), p.minM);
    return {
      wM, hM,
      center: f.at(fx + Math.sign(dx - fx || 1) * wM / 2,
                   fy + Math.sign(dy - fy || 1) * hM / 2) as [number, number],
    };
  }

  // Rotate: the handle's bearing about the plate centre, less the model's own
  // rotation — the plate angle is stored relative to the outline.
  function plateRotate(): PlateEdit {
    const p = plateRef.current!;
    const P = plateRotRef.current!.getLatLng();
    const mlon = M_PER_DEG_LON * Math.cos(p.center[0] * RAD);
    const vx = (P.lng - p.center[1]) * mlon;
    const vy = (P.lat - p.center[0]) * M_PER_DEG_LAT;
    let deg = Math.round(Math.atan2(vy, vx) / RAD - 90 - rotRef.current);
    deg = ((deg % 360) + 360) % 360;
    return { deg: deg > 180 ? deg - 360 : deg };
  }

  useEffect(() => {
    const map = mapRef.current!;
    if (!plate || !bbox) {
      for (const ref of [plateBoxRef, plateTetherRef] as const) {
        ref.current?.remove(); ref.current = undefined;
      }
      for (const ref of [plateMkRef, plateSwRef, plateNeRef, plateRotRef] as const) {
        ref.current?.remove(); ref.current = undefined;
      }
      return;
    }
    if (!plateBoxRef.current) {
      plateBoxRef.current = L.polygon([], {
        color: PLATE_COLOR, weight: 2, fillOpacity: 0.35, interactive: false,
      }).addTo(map);
      plateTetherRef.current = L.polyline([], {
        color: PLATE_COLOR, weight: 1, dashArray: "2 4", interactive: false,
      }).addTo(map);
      const mk = (icon: L.DivIcon) =>
        L.marker([0, 0], { icon, draggable: true, autoPan: true }).addTo(map);
      plateMkRef.current = mk(plateIcon);
      plateSwRef.current = mk(plateCornerIcon);
      plateNeRef.current = mk(plateCornerIcon);
      plateRotRef.current = mk(plateRotateIcon);

      const wire = (m: L.Marker, calc: () => PlateEdit) => {
        m.on("drag", () => redrawPlate(plateFrom(calc()), m));
        m.on("dragend", () => {
          const next = plateFrom(calc());
          redrawPlate(next);   // snap the handles even if the state is unchanged
          const { center, wM, hM, deg } = next;
          cbPlateRef.current?.({ center, wM, hM, deg });
        });
      };
      wire(plateMkRef.current, () => {
        const p = plateMkRef.current!.getLatLng();
        return { center: [p.lat, p.lng] };
      });
      wire(plateSwRef.current, () => plateResize(plateSwRef.current!, plateNeRef.current!));
      wire(plateNeRef.current, () => plateResize(plateNeRef.current!, plateSwRef.current!));
      wire(plateRotRef.current, plateRotate);
    }
    redrawPlate(plate);
  }, [plate, bbox, shape, rotation]);

  // Corner handles follow the size/scale lock: with the real-world extent
  // locked the frame can only pan and rotate. (bbox dep: the markers are
  // created just above once the first bbox arrives.)
  useEffect(() => {
    for (const m of [swRef.current, neRef.current]) {
      if (!m) continue;
      if (resizable) { m.dragging?.enable(); m.setOpacity(1); }
      else { m.dragging?.disable(); m.setOpacity(0.3); }
    }
  }, [resizable, bbox]);

  return <div ref={divRef} style={{ width: "100%", height: "100%" }} />;
}

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

// OSM slippy map showing the GPX track, with the model outline (rect / square /
// regular hexagon, rotatable) the user edits with three handles: corner ■ =
// resize, ● above the top edge = rotate, centre ✥ = move.
export function MapPicker({ points, bbox, shape, rotation, onBboxChange, onRotationChange }: {
  points: [number, number][]; // [lat, lon] in track order
  bbox: Bbox | null;          // unrotated shape extents
  shape: Shape;
  rotation: number;           // deg CCW
  onBboxChange: (bb: Bbox) => void;
  onRotationChange: (deg: number) => void;
}) {
  const divRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map>();
  const lineRef = useRef<L.Polyline>();
  const outlineRef = useRef<L.Polygon>();
  const frameRef = useRef<L.Polygon>();   // dashed bbox, shown for hex
  const tetherRef = useRef<L.Polyline>(); // top edge -> rotate handle
  const swRef = useRef<L.Marker>();
  const neRef = useRef<L.Marker>();
  const mvRef = useRef<L.Marker>();
  const rotMarkRef = useRef<L.Marker>();

  // Latest values for use inside Leaflet event handlers.
  const bboxRef = useRef(bbox); bboxRef.current = bbox;
  const shapeRef = useRef(shape); shapeRef.current = shape;
  const rotRef = useRef(rotation); rotRef.current = rotation;
  const cbBoxRef = useRef(onBboxChange); cbBoxRef.current = onBboxChange;
  const cbRotRef = useRef(onRotationChange); cbRotRef.current = onRotationChange;

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
    lineRef.current = L.polyline(points, { color: "#dc4628", weight: 3 }).addTo(map);
    map.fitBounds(lineRef.current.getBounds().pad(0.15));
  }, [points]);

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

  return <div ref={divRef} style={{ width: "100%", height: "100%" }} />;
}

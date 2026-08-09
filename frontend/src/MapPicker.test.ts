import { describe, expect, it } from "vitest";
import {
  M_PER_DEG_LAT, M_PER_DEG_LON, clampPlate, extentMeters, fitBbox, freeSpot,
  normalizeBbox, scaleBbox, spanMeters, type Bbox, type Shape,
} from "./MapPicker";

// The outline the user drags is the model's own extent, so its geometry is a
// contract with the backend: a "square" or "regular hexagon" has to be regular
// in printed millimetres (backend region.half_extents_m), not in degrees.

const RAD = Math.PI / 180;
const SQ3 = Math.sqrt(3);
const BB: Bbox = [139.0, 35.0, 139.02, 35.01];

// The bbox in local metres about its own centre.
function half(bb: Bbox): { hw: number; hh: number } {
  const clat = (bb[1] + bb[3]) / 2;
  return {
    hw: ((bb[2] - bb[0]) / 2) * M_PER_DEG_LON * Math.cos(clat * RAD),
    hh: ((bb[3] - bb[1]) / 2) * M_PER_DEG_LAT,
  };
}

const centre = (bb: Bbox): [number, number] =>
  [(bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2];

// Is [lat, lon] inside the outline? Both shapes are convex and centre-symmetric.
// The clamp puts a pushed plate *exactly* on the edge, and the plate's frame
// takes metres-per-degree at its own latitude while the outline's takes them at
// the model centre's — so allow the decimetre those two disagree by.
function inside(pt: [number, number], bb: Bbox, shape: Shape, rot: number) {
  const { hw, hh } = half(bb);
  const [clat, clon] = centre(bb);
  const mlon = M_PER_DEG_LON * Math.cos(clat * RAD);
  const th = rot * RAD;
  const dx = (pt[1] - clon) * mlon, dy = (pt[0] - clat) * M_PER_DEG_LAT;
  const x = dx * Math.cos(th) + dy * Math.sin(th);
  const y = -dx * Math.sin(th) + dy * Math.cos(th);
  const eps = 0.1;   // metres
  if (shape === "hex") {
    const d = (SQ3 / 2) * Math.min(hw, (2 * hh) / SQ3);
    return [30, 90, 150].every((a) =>
      Math.abs(x * Math.cos(a * RAD) + y * Math.sin(a * RAD)) <= d + eps);
  }
  const a = shape === "square" ? Math.min(hw, hh) : NaN;
  const [lx, ly] = shape === "square" ? [a, a] : [hw, hh];
  return Math.abs(x) <= lx + eps && Math.abs(y) <= ly + eps;
}

describe("normalizeBbox", () => {
  it("leaves a rectangle alone", () => {
    expect(normalizeBbox(BB, "rect")).toEqual(BB);
  });

  it("holds the backend's 0.001 deg floor on either side", () => {
    const [w, s, e, n] = normalizeBbox([139.0, 35.0, 139.00001, 35.00001], "rect");
    expect(e - w).toBeCloseTo(1e-3, 9);
    expect(n - s).toBeCloseTo(1e-3, 9);
  });

  it("makes a square square in metres, not in degrees", () => {
    const { hw, hh } = half(normalizeBbox(BB, "square"));
    expect(hw).toBeCloseTo(hh, 6);
  });

  it("gives a hexagon its flat-top aspect", () => {
    const { hw, hh } = half(normalizeBbox(BB, "hex"));
    expect(hh).toBeCloseTo((SQ3 / 2) * hw, 6);
  });

  it("grows rather than shrinks, so a drag never loses area", () => {
    const { hw: hw0, hh: hh0 } = half(BB);
    const { hw, hh } = half(normalizeBbox(BB, "square"));
    expect(Math.max(hw, hh)).toBeCloseTo(Math.max(hw0, hh0), 6);
  });

  it("is stable under re-application", () => {
    // It runs on every drag, so a normalisation that crept would drift the
    // outline a little further out with every mouse move.
    for (const shape of ["rect", "square", "hex"] as Shape[]) {
      const once = normalizeBbox(BB, shape);
      normalizeBbox(once, shape).forEach((v, i) => expect(v).toBeCloseTo(once[i], 9));
    }
  });

  it("keeps the centre put", () => {
    for (const shape of ["square", "hex"] as Shape[]) {
      const out = normalizeBbox(BB, shape);
      expect(centre(out)[0]).toBeCloseTo(centre(BB)[0], 9);
      expect(centre(out)[1]).toBeCloseTo(centre(BB)[1], 9);
    }
  });
});

describe("fitBbox", () => {
  const track: [number, number][] = [
    [35.001, 139.001], [35.004, 139.006], [35.008, 139.003], [35.006, 139.012],
  ];

  it("has nothing to fit around an empty track", () => {
    expect(fitBbox([], "rect", 0)).toBeNull();
  });

  it.each([
    ["rect", 0], ["square", 0], ["hex", 0],
    ["rect", 37], ["square", 37], ["hex", 37],
  ] as [Shape, number][])("encloses every point (%s at %s deg)", (shape, rot) => {
    const bb = fitBbox(track, shape, rot)!;
    for (const p of track) expect(inside(p, bb, shape, rot)).toBe(true);
  });

  it("leaves a margin rather than hugging the track", () => {
    const bb = fitBbox(track, "rect", 0)!;
    const span = Math.max(...track.map((p) => p[1])) -
                 Math.min(...track.map((p) => p[1]));
    expect(bb[2] - bb[0]).toBeGreaterThan(span * 1.15);
  });

  it("still has area when the track is a single point", () => {
    const bb = fitBbox([[35.0, 139.0]], "rect", 0)!;
    expect(bb[2] - bb[0]).toBeCloseTo(1e-3, 9);   // the backend's floor
    expect(bb[3] - bb[1]).toBeCloseTo(1e-3, 9);
  });
});

describe("spanMeters", () => {
  // What size_mm maps onto — it must mirror the backend's region.span_m, so
  // the scale the panel shows is the scale that prints.
  it("takes the rectangle's long edge", () => {
    const { hw, hh } = half(BB);
    expect(spanMeters(BB, "rect")).toBeCloseTo(2 * Math.max(hw, hh), 6);
  });

  it("inscribes the square and the hexagon in the bbox", () => {
    const { hw, hh } = half(BB);
    expect(spanMeters(BB, "square")).toBeCloseTo(2 * Math.min(hw, hh), 6);
    expect(spanMeters(BB, "hex")).toBeCloseTo(2 * Math.min(hw, (2 * hh) / SQ3), 6);
  });

  it("agrees with the backend on a known bbox", () => {
    // The same bbox and the same expected metres as backend
    // tests/test_region.py::test_the_span_matches_the_frontends — the two
    // sides must not drift apart, or the printed scale stops matching the
    // number the panel showed.
    expect(spanMeters([139.0, 35.0, 139.02, 35.01], "rect")).toBeCloseTo(1823.6487, 3);
    expect(extentMeters([139.0, 35.0, 139.02, 35.01])[1]).toBeCloseTo(1105.4, 6);
  });
});

describe("scaleBbox", () => {
  it("rescales about the centre", () => {
    const out = scaleBbox(BB, 2);
    expect(centre(out)).toEqual(centre(BB));
    expect(out[2] - out[0]).toBeCloseTo(2 * (BB[2] - BB[0]), 12);
  });

  it("keeps the aspect, so a square stays square", () => {
    const bb = normalizeBbox(BB, "square");
    const { hw, hh } = half(scaleBbox(bb, 0.5));
    expect(hw).toBeCloseTo(hh, 6);
  });
});

describe("clampPlate", () => {
  const W = 300, H = 120;      // plate size in metres

  it.each(["rect", "square", "hex"] as Shape[])(
    "pulls a plate outside the %s outline back in", (shape) => {
      const bb = normalizeBbox(BB, shape);
      const [clat, clon] = centre(bb);
      const out = clampPlate([clat + 0.02, clon + 0.05], bb, shape, 0, W, H, 0);
      for (const corner of plateCorners(out, W, H, 0)) {
        expect(inside(corner, bb, shape, 0)).toBe(true);
      }
    });

  it("keeps the whole rectangle in, not just its centre", () => {
    const bb = normalizeBbox(BB, "rect");
    const out = clampPlate([35.5, 139.5], bb, "rect", 0, W, H, 0);
    for (const corner of plateCorners(out, W, H, 0)) {
      expect(inside(corner, bb, "rect", 0)).toBe(true);
    }
  });

  it("counts the plate's own angle", () => {
    const bb = normalizeBbox(BB, "rect");
    const flat = clampPlate([35.5, 139.5], bb, "rect", 0, W, H, 0);
    const turned = clampPlate([35.5, 139.5], bb, "rect", 0, W, H, 90);
    expect(turned[1]).not.toBeCloseTo(flat[1], 9);
    for (const corner of plateCorners(turned, W, H, 90)) {
      expect(inside(corner, bb, "rect", 0)).toBe(true);
    }
  });

  it("leaves a plate that already fits where it is", () => {
    const bb = normalizeBbox(BB, "rect");
    const [clat, clon] = centre(bb);
    const out = clampPlate([clat, clon], bb, "rect", 0, W, H, 0);
    expect(out[0]).toBeCloseTo(clat, 9);
    expect(out[1]).toBeCloseTo(clon, 9);
  });

  it("centres a plate wider than the model instead of pushing it off", () => {
    // The backend cuts the overhang flush with the model edge; a clamp that
    // still tried to fit it would slide it off the map entirely.
    const bb = normalizeBbox(BB, "rect");
    const [clat, clon] = centre(bb);
    const out = clampPlate([clat + 0.004, clon], bb, "rect", 0, 99_000, 99_000, 0);
    expect(out[0]).toBeCloseTo(clat, 6);
    expect(out[1]).toBeCloseTo(clon, 6);
  });
});

describe("freeSpot", () => {
  const W = 200, H = 80;

  it("puts the plate at the centre when there is no track", () => {
    const bb = normalizeBbox(BB, "rect");
    const [clat, clon] = centre(bb);
    const out = freeSpot([], bb, "rect", 0, W, H, 0);
    expect(out[0]).toBeCloseTo(clat, 9);
    expect(out[1]).toBeCloseTo(clon, 9);
  });

  it("moves the plate off a track running through the middle", () => {
    const bb = normalizeBbox(BB, "rect");
    const [clat] = centre(bb);
    const track: [number, number][] = Array.from({ length: 40 }, (_, i) => [
      clat, bb[0] + ((bb[2] - bb[0]) * i) / 39,
    ]);
    const out = freeSpot(track, bb, "rect", 0, W, H, 0);
    const dy = Math.abs(out[0] - clat) * M_PER_DEG_LAT;
    expect(dy).toBeGreaterThan(H / 2);
  });

  it("returns a spot the plate actually fits in", () => {
    const bb = normalizeBbox(BB, "hex");
    const track: [number, number][] = [[35.004, 139.008], [35.006, 139.012]];
    const out = freeSpot(track, bb, "hex", 20, W, H, 20);
    for (const corner of plateCorners(out, W, H, 20 + 20)) {
      expect(inside(corner, bb, "hex", 20)).toBe(true);
    }
  });
});

// The four corners of a plate centred at [lat, lon], turned by `deg`.
function plateCorners(
  center: [number, number], wM: number, hM: number, deg: number,
): [number, number][] {
  const th = deg * RAD, c = Math.cos(th), s = Math.sin(th);
  const mlon = M_PER_DEG_LON * Math.cos(center[0] * RAD);
  return ([[-1, -1], [1, -1], [1, 1], [-1, 1]] as const).map(([sx, sy]) => {
    const x = (sx * wM) / 2, y = (sy * hM) / 2;
    return [
      center[0] + (x * s + y * c) / M_PER_DEG_LAT,
      center[1] + (x * c - y * s) / mlon,
    ] as [number, number];
  });
}

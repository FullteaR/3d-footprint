import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

export type Bbox = [number, number, number, number]; // west, south, east, north

// Corner drag handles as plain divs (Leaflet's default marker PNGs don't
// survive bundling, and we want a resize-grip look anyway).
const handleIcon = L.divIcon({
  className: "",
  html: '<div style="width:14px;height:14px;background:#fff;border:2px solid #d33;border-radius:3px;box-shadow:0 1px 3px #0006;cursor:nwse-resize"></div>',
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

// OSM slippy map showing the GPX track, with the model bbox as a rectangle
// the user resizes by dragging its SW/NE corner handles.
export function MapPicker({ points, bbox, onBboxChange }: {
  points: [number, number][]; // [lat, lon] in track order
  bbox: Bbox | null;
  onBboxChange: (bb: Bbox) => void;
}) {
  const divRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map>();
  const lineRef = useRef<L.Polyline>();
  const rectRef = useRef<L.Rectangle>();
  const swRef = useRef<L.Marker>();
  const neRef = useRef<L.Marker>();
  const cbRef = useRef(onBboxChange);
  cbRef.current = onBboxChange;

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

  // Bbox rectangle + corner handles. Dragging emits onBboxChange (on dragend,
  // normalized so west<east / south<north even if the corners crossed);
  // external bbox changes (reset button) just move the layers — no emit, so
  // there is no update loop.
  useEffect(() => {
    const map = mapRef.current!;
    if (!bbox) {
      rectRef.current?.remove();
      swRef.current?.remove();
      neRef.current?.remove();
      rectRef.current = swRef.current = neRef.current = undefined;
      return;
    }
    const [w, s, e, n] = bbox;
    if (!rectRef.current) {
      rectRef.current = L.rectangle([[s, w], [n, e]], {
        color: "#d33", weight: 2, fillOpacity: 0.06, interactive: false,
      }).addTo(map);
      swRef.current = L.marker([s, w], { icon: handleIcon, draggable: true, autoPan: true }).addTo(map);
      neRef.current = L.marker([n, e], { icon: handleIcon, draggable: true, autoPan: true }).addTo(map);
      const sync = (emit: boolean) => {
        const a = swRef.current!.getLatLng();
        const b = neRef.current!.getLatLng();
        const bb: Bbox = [
          Math.min(a.lng, b.lng), Math.min(a.lat, b.lat),
          Math.max(a.lng, b.lng), Math.max(a.lat, b.lat),
        ];
        rectRef.current!.setBounds([[bb[1], bb[0]], [bb[3], bb[2]]]);
        if (emit) cbRef.current(bb);
      };
      for (const h of [swRef.current, neRef.current]) {
        h.on("drag", () => sync(false));
        h.on("dragend", () => sync(true));
      }
    } else {
      rectRef.current.setBounds([[s, w], [n, e]]);
      swRef.current!.setLatLng([s, w]);
      neRef.current!.setLatLng([n, e]);
    }
  }, [bbox]);

  return <div ref={divRef} style={{ width: "100%", height: "100%" }} />;
}

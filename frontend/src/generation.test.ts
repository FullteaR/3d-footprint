import { describe, expect, it, vi, afterEach } from "vitest";
import { formKey, waitForJob } from "./generation";
import { disposeModel } from "./dispose";
import * as THREE from "three";

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

describe("preview identity", () => {
  it("matches every model setting, but permits another export format", () => {
    const file = new File(["a"], "route.gpx");
    const form = new FormData();
    form.append("file", file);
    form.append("size_mm", "120");
    form.append("fmt", "glb");
    const key = formKey(form);
    form.set("fmt", "stl");
    expect(formKey(form)).toBe(key);
    form.set("size_mm", "121");
    expect(formKey(form)).not.toBe(key);
    form.set("size_mm", "120");
    form.set("file", new File(["b"], "route.gpx"));
    expect(formKey(form)).not.toBe(key);
  });
});

describe("job polling", () => {
  it("waits for ready artifacts and retains partial-data warnings", async () => {
    vi.useFakeTimers();
    const warning = { source: "buildings", reason: "fetch_failed" };
    const fetcher = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "job", status: "running", stage: "terrain" })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "job", status: "succeeded", warnings: [warning] })));
    vi.stubGlobal("fetch", fetcher);
    const update = vi.fn();
    const pending = waitForJob("job", update, new AbortController().signal);
    await vi.advanceTimersByTimeAsync(1000);
    expect((await pending).warnings).toEqual([warning]);
    expect(update).toHaveBeenCalledTimes(2);
    expect(fetcher.mock.calls.every(([url]) => url === "/api/jobs/job")).toBe(true);
  });

  it("stops polling after cancellation", async () => {
    const controller = new AbortController();
    controller.abort();
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    await expect(waitForJob("job", vi.fn(), controller.signal)).rejects.toBeDefined();
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("reports a failed generation without requesting model files", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      status: "failed", detail: "deadline exceeded",
    }))));
    await expect(waitForJob("job", vi.fn(), new AbortController().signal)).rejects.toThrow("deadline exceeded");
  });
});

it("disposes shared GPU resources once when an old model is removed", () => {
  const geometry = new THREE.BoxGeometry();
  const texture = new THREE.Texture();
  const material = new THREE.MeshStandardMaterial({ map: texture });
  const model = new THREE.Group();
  model.add(new THREE.Mesh(geometry, material), new THREE.Mesh(geometry, [material]));
  const geomDispose = vi.spyOn(geometry, "dispose");
  const materialDispose = vi.spyOn(material, "dispose");
  const textureDispose = vi.spyOn(texture, "dispose");
  disposeModel(model);
  expect(geomDispose).toHaveBeenCalledTimes(1);
  expect(materialDispose).toHaveBeenCalledTimes(1);
  expect(textureDispose).toHaveBeenCalledTimes(1);
});

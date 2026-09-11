import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi, type Mock } from "vitest";
import { App } from "./App";
import { TEXT } from "./i18n";

vi.mock("./Preview", () => ({ Preview: () => null }));
vi.mock("./MapPicker", async (original) => ({
  ...await original<typeof import("./MapPicker")>(), MapPicker: () => null,
}));

let root: Root;
let mount: HTMLDivElement;
let fetcher: Mock<(url: string, options?: RequestInit) => Promise<Response>>;
const key = "1234567890abcdef1234567890abcdef";

function button(text: string): HTMLButtonElement {
  const element = Array.from(mount.querySelectorAll("button")).find((b) => b.textContent === text);
  if (!element) throw new Error(`Missing button: ${text}`);
  return element;
}

beforeEach(async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  localStorage.setItem("3dfp.lang", "ja");
  vi.spyOn(crypto, "randomUUID").mockReturnValue("12345678-90ab-cdef-1234-567890abcdef");
  fetcher = vi.fn(async (url: string, options?: RequestInit) => {
    if (url === "/api/health") return Response.json({ status: "ok" });
    if (options?.method === "DELETE") return new Response(null, { status: 204 });
    if (url === "/api/generate") return Response.json({ id: key }, { status: 202 });
    if (url === `/api/jobs/${key}`) return Response.json({
      id: key, status: "succeeded", stage: "succeeded", expires_at: Date.now() / 1000 + 3600,
      warnings: [{ source: "buildings", reason: "fetch_failed" }],
    });
    if (url.includes("/files/")) return new Response(new Uint8Array([1, 2, 3]));
    throw new Error(`Unexpected request: ${url}`);
  });
  vi.stubGlobal("fetch", fetcher);
  vi.stubGlobal("URL", Object.assign(URL, {
    createObjectURL: vi.fn(() => "blob:download"), revokeObjectURL: vi.fn(),
  }));
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  mount = document.createElement("div");
  document.body.append(mount);
  root = createRoot(mount);
  await act(async () => root.render(createElement(App)));
  const file = new File(["gpx"], "route.gpx");
  Object.defineProperty(file, "text", { value: async () =>
    '<gpx><trk><trkseg><trkpt lat="35.003" lon="139.003"/><trkpt lat="35.007" lon="139.007"/></trkseg></trk></gpx>' });
  const input = mount.querySelector<HTMLInputElement>('input[type="file"]')!;
  Object.defineProperty(input, "files", { value: [file] });
  await act(async () => { input.dispatchEvent(new Event("change", { bubbles: true })); });
});

afterEach(async () => {
  await act(async () => root.unmount());
  mount.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("downloads the previewed job, then refuses download after a model setting changes", async () => {
  expect(button(TEXT.ja.download).disabled).toBe(true);
  await act(async () => { button(TEXT.ja.create).click(); });
  expect(button(TEXT.ja.download).disabled).toBe(false);
  expect(mount.textContent).toContain("一部の取得・解析に失敗");
  await act(async () => { button(TEXT.ja.download).click(); });
  expect(fetcher.mock.calls.filter(([url]) => url === "/api/generate")).toHaveLength(1);
  expect(fetcher.mock.calls.some(([url]) => url === `/api/jobs/${key}/files/3mf`)).toBe(true);
  const checkbox = mount.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
  await act(async () => checkbox.click());
  expect(button(TEXT.ja.download).disabled).toBe(true);
  expect(mount.textContent).toContain(TEXT.ja.previewStale);
});

it("keeps a completion for old settings marked stale if the user edits during generation", async () => {
  let complete!: (value: Response) => void;
  const pending = new Promise<Response>((resolve) => { complete = resolve; });
  const normalFetch = fetcher.getMockImplementation()!;
  fetcher.mockImplementation((url, options) => url === `/api/jobs/${key}` ? pending : normalFetch(url, options));
  await act(async () => { button(TEXT.ja.create).click(); });
  const checkbox = mount.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
  await act(async () => checkbox.click());
  await act(async () => complete(Response.json({
    id: key, status: "succeeded", stage: "succeeded", expires_at: Date.now() / 1000 + 3600, warnings: [],
  })));
  expect(button(TEXT.ja.download).disabled).toBe(true);
  expect(mount.textContent).toContain(TEXT.ja.previewStale);
});

it("waits for cancellation even if generation completes before the DELETE response", async () => {
  let complete!: (value: Response) => void;
  let cancelled!: (value: Response) => void;
  const pendingJob = new Promise<Response>((resolve) => { complete = resolve; });
  const pendingDelete = new Promise<Response>((resolve) => { cancelled = resolve; });
  const normalFetch = fetcher.getMockImplementation()!;
  fetcher.mockImplementation((url, options) => {
    if (options?.method === "DELETE") return pendingDelete;
    if (url === `/api/jobs/${key}`) return pendingJob;
    return normalFetch(url, options);
  });
  await act(async () => { button(TEXT.ja.create).click(); });
  await act(async () => { button(TEXT.ja.cancel).click(); });
  expect(button(TEXT.ja.cancel).disabled).toBe(true);
  await act(async () => complete(Response.json({
    id: key, status: "succeeded", stage: "succeeded", expires_at: Date.now() / 1000 + 3600, warnings: [],
  })));
  expect(button(TEXT.ja.create).disabled).toBe(true);
  expect(button(TEXT.ja.download).disabled).toBe(true);
  await act(async () => cancelled(new Response(null, { status: 204 })));
  expect(button(TEXT.ja.create).disabled).toBe(false);
  expect(button(TEXT.ja.download).disabled).toBe(true);
  expect(mount.textContent).toContain(TEXT.ja.stCancelled);
});

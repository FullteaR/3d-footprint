import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Dev: proxy /api to the FastAPI backend so the frontend can call it directly.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
  },
  test: {
    // MapPicker imports leaflet, which touches `window` as it loads, and
    // i18n reads localStorage/navigator — so the tests need a document.
    environment: "jsdom",
    include: ["src/**/*.test.ts"],
  },
});

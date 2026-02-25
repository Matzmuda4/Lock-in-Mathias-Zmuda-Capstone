import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Tauri expects a fixed port and strict mode
  server: {
    port: 5173,
    strictPort: true,
    watch: {
      // Don't watch Rust source — Tauri handles its own rebuild
      ignored: ["**/src-tauri/**"],
    },
  },
  // Expose VITE_ and TAURI_ env vars to the frontend
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    // Tauri supports modern browsers only
    target: ["es2021", "chrome100", "safari15"],
    minify: !process.env.TAURI_ENV_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
  },
  // Vitest config lives here so we don't need a separate file
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
  },
});

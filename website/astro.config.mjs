import { defineConfig } from "astro/config";
import { createReadStream } from "node:fs";
import { stat, cp, access } from "node:fs/promises";
import { resolve, sep, extname } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));
const cacheRoot = resolve(root, "cache/viewer");
function serveCache(server) {
  const prefix = `${server.config.base.replace(/\/?$/, "/")}viewer-data/`;
  server.middlewares.use(async (req, res, next) => {
    if (!req.url?.startsWith(prefix)) return next();
    try {
      const name = decodeURIComponent(new URL(req.url, "http://localhost").pathname.slice(prefix.length));
      const path = resolve(cacheRoot, name);
      if (!path.startsWith(cacheRoot + sep) || ![".json", ".bin"].includes(extname(path))) {
        res.statusCode = 403; res.end(); return;
      }
      const info = await stat(path);
      if (!info.isFile()) throw new Error("Not a cache file");
      res.setHeader("Content-Type", path.endsWith(".json") ? "application/json" : "application/octet-stream");
      res.setHeader("Content-Length", info.size);
      res.setHeader("Cache-Control", "no-cache");
      if (req.method === "HEAD") { res.end(); return; }
      createReadStream(path).on("error", () => res.destroy()).pipe(res);
    } catch {
      res.statusCode = 404; res.end("Cache file missing. Run pnpm cache:build first.");
    }
  });
}
export default defineConfig({
  output: "static",
  site: process.env.PAGES_ORIGIN,
  base: `${(process.env.PAGES_BASE_PATH || "").replace(/\/$/, "")}/`,
  integrations: [{
    name: "parte-viewer-data",
    hooks: {
      "astro:build:start": async () => {
        try { await access(resolve(cacheRoot, "manifest.json")); }
        catch { throw new Error("Generate the viewer cache with pnpm cache:build before building the website."); }
      },
      "astro:build:done": async ({ dir }) => {
        await cp(cacheRoot, new URL("viewer-data/", dir), { recursive: true });
      },
    },
  }],
  build: { assets: "assets" },
  vite: {
    server: { watch: { ignored: ["**/cache/**"] } },
    plugins: [{ name: "parte-local-cache", configureServer: serveCache }],
    build: { chunkSizeWarningLimit: 600 },
  },
});

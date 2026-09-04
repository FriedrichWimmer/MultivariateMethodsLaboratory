import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * The production bundle is served from a sub-path on GitHub Pages
 * (https://<user>.github.io/<repo>/), so asset URLs must be prefixed with it.
 * The deploy workflow passes BASE_PATH; local `npm run dev` and `npm run build`
 * fall back to "/" so nothing changes for local use.
 */
const base = process.env.BASE_PATH ?? '/';

export default defineConfig({
  base,
  plugins: [react()],
  server: { port: 5173, open: false },
  build: {
    chunkSizeWarningLimit: 4000,
  },
});

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    proxy: {
      '/upload':      'http://localhost:5000',
      '/files':       'http://localhost:5000',
      '/download-url': 'http://localhost:5000',
    }
  },
  plugins: [react()],
});
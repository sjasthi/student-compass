import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  server: {
    proxy: {
      '/upload':       'http://localhost:5000',
      '/files':        'http://localhost:5000',
      '/download-url': 'http://localhost:5000',
      '/query':        'http://localhost:5000',
      '/sync':         'http://localhost:5000',
      '/health':       'http://localhost:5000',
    },
  },
  plugins: [react()],
});

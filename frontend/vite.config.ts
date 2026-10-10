import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 5173,
    proxy: {
      '/jobs': 'http://127.0.0.1:8020',
      '/api': 'http://127.0.0.1:8020',
    },
  },
})

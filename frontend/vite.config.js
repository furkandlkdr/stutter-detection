import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Kök domain için '/'; alt yol kullanılacaksa değiştirilebilir.
export default defineConfig({
  base: '/',
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: 5173,
    allowedHosts: ['.ngrok-free.dev', '.furkan.software', 'localhost'],
  },
})

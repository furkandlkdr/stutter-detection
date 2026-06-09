import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { VitePWA } from 'vite-plugin-pwa'

// Kök domain için '/'; alt yol kullanılacaksa değiştirilebilir.
export default defineConfig({
  base: '/',
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      // Geliştirmede de SW'ı görünür kılarak test imkânı sağlar.
      // Üretimde `registerType: 'autoUpdate'` ile yeni build sessizce devreye alınır.
      devOptions: {
        enabled: true,
        type: 'module',
      },
      registerType: 'autoUpdate',

      // includeAssets içindeki dosyalar precache'e eklenir.
      includeAssets: [
        'favicon.ico',
        'icons/icon-192.png',
        'icons/icon-512.png',
        'icons/icon-maskable-512.png',
        'apple-touch-icon.png',
      ],

      // PWA manifesti. Sade tutuldu; gerçek üretim ikonları sonradan değiştirilir.
      manifest: {
        name: 'Kekemelik Analiz Sistemi',
        short_name: 'Kekemelik AI',
        description:
          'Yapay zekâ destekli kekemelik tespit sistemi. Ses yükle ya da mikrofondan kayıt yap.',
        lang: 'tr',
        dir: 'ltr',
        start_url: '/',
        scope: '/',
        display: 'standalone',
        orientation: 'portrait',
        background_color: '#f8fafc',
        theme_color: '#2563eb',
        categories: ['health', 'medical', 'utilities', 'productivity'],
        icons: [
          {
            src: 'icons/icon-192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any',
          },
          {
            src: 'icons/icon-512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any',
          },
          {
            src: 'icons/icon-maskable-512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable',
          },
        ],
      },

      // Workbox runtime cache stratejileri.
      // - app shell: precache (manifest tarafından yönetilir)
      // - /health: staleWhileRevalidate (offline'da son bilinen durumu gösterir)
      // - /analyze: SW tarafından proxy'lenmez (multipart/form-data + CORS sözleşmesi bozulur)
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff,woff2}'],
        cleanupOutdatedCaches: true,
        clientsClaim: true,
        skipWaiting: true,
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [/^\/api/, /^\/health/, /^\/analyze/],
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.endsWith('/health'),
            handler: 'StaleWhileRevalidate',
            options: {
              cacheName: 'api-health',
              expiration: { maxEntries: 8, maxAgeSeconds: 60 * 60 * 24 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
          {
            // Backend'in diğer GET uçlarını network-first yap ki eski sonuçlar anlık göstermesin.
            urlPattern: ({ url, request }) =>
              request.method === 'GET' &&
              (url.pathname.startsWith('/api/') || url.pathname === '/'),
            handler: 'NetworkFirst',
            options: {
              cacheName: 'api-get',
              networkTimeoutSeconds: 4,
              expiration: { maxEntries: 32, maxAgeSeconds: 60 * 60 * 6 },
            },
          },
          {
            // Statik font/img varlıkları için cache-first.
            urlPattern: ({ request }) =>
              ['style', 'script', 'worker', 'image', 'font'].includes(
                request.destination
              ),
            handler: 'CacheFirst',
            options: {
              cacheName: 'static-assets',
              expiration: { maxEntries: 64, maxAgeSeconds: 60 * 60 * 24 * 30 },
            },
          },
        ],
      },
    }),
  ],
  server: {
    host: true,
    port: 5173,
    allowedHosts: ['.ngrok-free.dev', '.furkan.software', 'localhost'],
  },
})

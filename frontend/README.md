# Stutter Detection — Frontend

React + Vite + Tailwind CSS tabanlı arayüz. Backend API ile konuşur.
**PWA desteklidir** — mobil cihazlarda "Ana Ekrana Ekle" ile yüklenebilir.

## Kurulum

```bash
cd frontend
npm install
```

## Geliştirme (dev server)

```bash
npm run dev
# http://localhost:5173
```

## Üretim derlemesi (GitHub Pages için)

```bash
npm run build
# Çıktı: dist/ klasörü (manifest, sw.js, ikonlar dahil)
```

`vite.config.js` içindeki `base` alanı GitHub Pages altındaki alt yolu temsil eder
(ör. `/stutter-detection/`). GitHub Pages'e deploy ederken repo adına göre güncelleyin.

## PWA Özellikleri

Uygulama `vite-plugin-pwa` ile PWA olarak derlenir. Build çıktısı:

- `dist/manifest.webmanifest` — uygulama adı, ikonlar, tema rengi, `display: standalone`.
- `dist/sw.js` + `dist/workbox-*.js` — Workbox tabanlı Service Worker.
- `dist/icons/` — 192×192, 512×512, maskable 512×512 ve 180×180 Apple touch ikonları.

### Cache Stratejisi

| Rota türü | Strateji | Açıklama |
| --- | --- | --- |
| App shell (HTML/JS/CSS/img/font) | **Precache** (build sırasında) | Offline'da sayfa açılır |
| `GET /health` | **StaleWhileRevalidate** | Çevrimdışı iken son bilinen sağlık durumu gösterilir |
| `GET /api/*`, `GET /` | **NetworkFirst** (4 sn timeout) | Taze veri, offline'da cache |
| `POST /analyze` | SW tarafından proxy'lenmez | Multipart form-data + CORS sözleşmesine dokunulmaz |

### Yerel olarak PWA test etme

```bash
npm run build && npm run preview
# http://127.0.0.1:4173
# DevTools > Application > Manifest / Service Workers
```

> Not: `localhost` üzerinde SW çalışır. Telefondan test etmek için `ngrok http 4173`
> veya GitHub Pages üzerinden HTTPS ile yayınlayın (Service Worker yalnızca
> güvenli bağlamda aktive olur).

### Gerçek ikon değiştirme

`public/icons/` altındaki dosyaları aynı isimle değiştirin:

- `icon-192.png` (192×192)
- `icon-512.png` (512×512)
- `icon-maskable-512.png` (512×512, **güvenli bölge ortada %40 alan**)
- `apple-touch-icon.png` (180×180)

Manifest'teki yollar `vite.config.js` → `includeAssets` ve `manifest.icons` üzerinden
güncellenebilir.

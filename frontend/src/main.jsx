import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { registerSW } from 'virtual:pwa-register'
import App from './App.jsx'
import './index.css'

// Service Worker kaydı — vite-plugin-pwa tarafından sağlanır.
// Üretimde yeni build olduğunda `autoUpdate` stratejisiyle SW sessizce
// yenilenir; ilk isteği SW halletmediği için anlık güncelleme deneyimi sağlanır.
const updateSW = registerSW({
  immediate: true,
  onRegisterError(error) {
    // Service Worker kaydı başarısız olursa (örn. çok eski tarayıcı) logla.
    // PWA zorunlu değil; uygulama yine de çalışır.
    // eslint-disable-next-line no-console
    console.warn('[PWA] Service Worker kaydı başarısız:', error)
  },
  onRegisteredSW(swUrl) {
    // eslint-disable-next-line no-console
    console.info('[PWA] Service Worker kayıtlı:', swUrl)
  },
  onNeedRefresh() {
    // Üretimde isteğe bağlı toast/banner gösterilebilir.
    // eslint-disable-next-line no-console
    console.info('[PWA] Yeni sürüm hazır; sayfayı yenileyince uygulanır.')
  },
  onOfflineReady() {
    // eslint-disable-next-line no-console
    console.info('[PWA] Çevrimdışı kullanıma hazır.')
  },
})
// updateSW referansını ileride manuel güncelleme tetiklemek için dışa açık bırakıyoruz.
if (typeof window !== 'undefined') {
  window.__pwaUpdateSW = updateSW
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>
)

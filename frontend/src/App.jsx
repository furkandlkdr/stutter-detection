import { useEffect, useMemo, useRef, useState } from 'react'

// --- Kalıcı ayarlar -------------------------------------------------------

const STORAGE_KEY = 'stutter_api_url'

function loadApiUrl() {
  try {
    return localStorage.getItem(STORAGE_KEY) || ''
  } catch {
    return ''
  }
}

function saveApiUrl(url) {
  try {
    if (url) localStorage.setItem(STORAGE_KEY, url)
    else localStorage.removeItem(STORAGE_KEY)
  } catch {
    /* yok say */
  }
}

function normalizeBase(url) {
  return (url || '').trim().replace(/\/+$/, '')
}

function formatClock(seconds) {
  const mm = Math.floor(seconds / 60).toString().padStart(2, '0')
  const ss = Math.floor(seconds % 60).toString().padStart(2, '0')
  return `${mm}:${ss}`
}

async function safeReadDetail(response) {
  try {
    const data = await response.json()
    return data?.detail || JSON.stringify(data)
  } catch {
    try {
      return await response.text()
    } catch {
      return ''
    }
  }
}

// --- Mikrofon kaydı: WAV dönüşümü ----------------------------------------
// MediaRecorder tarayıcıda webm/ogg/mp4 üretir; backend librosa ise WAV
// beklediği için burada ses verisini decode edip 16 kHz mono PCM'e çeviriyoruz.
const TARGET_SAMPLE_RATE = 16000

function pickSupportedMime() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ]
  if (typeof MediaRecorder === 'undefined') return ''
  for (const m of candidates) {
    if (MediaRecorder.isTypeSupported(m)) return m
  }
  return ''
}

function encodeWavMono(float32Samples, sampleRate) {
  const numChannels = 1
  const bitsPerSample = 16
  const dataSize = float32Samples.length * 2
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i))
    }
  }

  // RIFF başlığı
  writeString(0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeString(8, 'WAVE')
  // fmt alt bloğu
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8
  const blockAlign = (numChannels * bitsPerSample) / 8
  view.setUint32(28, byteRate, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, bitsPerSample, true)
  // data alt bloğu
  writeString(36, 'data')
  view.setUint32(40, dataSize, true)

  // PCM 16-bit örnekler
  let offset = 44
  for (let i = 0; i < float32Samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, float32Samples[i]))
    const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7fff
    view.setInt16(offset, intSample, true)
    offset += 2
  }
  return new Blob([view], { type: 'audio/wav' })
}

async function blobToMonoWav(blob) {
  const arrayBuffer = await blob.arrayBuffer()
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext
  if (!Ctx) {
    throw new Error('Tarayıcı OfflineAudioContext desteklemiyor.')
  }
  const tempCtx = new (window.AudioContext || window.webkitAudioContext)()
  let decoded
  try {
    decoded = await tempCtx.decodeAudioData(arrayBuffer.slice(0))
  } finally {
    tempCtx.close().catch(() => {})
  }
  const offline = new Ctx(
    1,
    Math.ceil(decoded.duration * TARGET_SAMPLE_RATE),
    TARGET_SAMPLE_RATE
  )
  const source = offline.createBufferSource()
  source.buffer = decoded
  source.connect(offline.destination)
  source.start(0)
  const rendered = await offline.startRendering()
  return encodeWavMono(rendered.getChannelData(0), rendered.sampleRate)
}

// --- Ana bileşen ----------------------------------------------------------

export default function App() {
  const [apiUrl, setApiUrl] = useState(loadApiUrl)
  const [urlDraft, setUrlDraft] = useState(loadApiUrl)
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [health, setHealth] = useState({ state: 'idle', message: '' })

  // Mikrofon kaydı state'leri
  const [isRecording, setIsRecording] = useState(false)
  const [recordingSeconds, setRecordingSeconds] = useState(0)
  const [converting, setConverting] = useState(false)
  const [micError, setMicError] = useState('')
  const [isSecure, setIsSecure] = useState(true)
  const [liveLevels, setLiveLevels] = useState([])

  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const tickRef = useRef(null)
  const startedAtRef = useRef(0)
  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)
  const animRef = useRef(null)

  const effectiveUrl = useMemo(() => normalizeBase(apiUrl), [apiUrl])

  useEffect(() => {
    saveApiUrl(apiUrl)
  }, [apiUrl])

  // URL kaydedilince backend sağlık kontrolü
  useEffect(() => {
    if (!effectiveUrl) {
      setHealth({ state: 'idle', message: '' })
      return
    }
    let cancelled = false
    const checkHealth = async () => {
      setHealth({ state: 'loading', message: 'Sunucu kontrol ediliyor…' })
      try {
        const res = await fetch(`${effectiveUrl}/health`, {
          method: 'GET',
          headers: { 'ngrok-skip-browser-warning': 'true' },
        })
        if (cancelled) return
        if (res.ok) {
          const data = await res.json()
          if (data?.ready) {
            setHealth({ state: 'ok', message: 'API hazır ✓' })
          } else {
            setHealth({
              state: 'warn',
              message: 'API yanıt verdi ancak model hazır değil',
            })
          }
        } else {
          setHealth({ state: 'error', message: `HTTP ${res.status}` })
        }
      } catch {
        if (cancelled) return
        setHealth({ state: 'error', message: 'Sunucuya ulaşılamıyor' })
      }
    }
    checkHealth()
    return () => {
      cancelled = true
    }
  }, [effectiveUrl])

  // Sayfa kapanırken medya stream'i serbest bırak
  useEffect(() => {
    return () => {
      if (tickRef.current) clearInterval(tickRef.current)
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  // HTTPS / localhost kontrolü — mobil mikrofon erişimi için gerekli
  useEffect(() => {
    try {
      const secure =
        window.isSecureContext === true ||
        window.location.protocol === 'https:' ||
        window.location.hostname === 'localhost' ||
        window.location.hostname === '127.0.0.1'
      setIsSecure(secure)
    } catch {
      setIsSecure(false)
    }
  }, [])

  // --- Dosya seçimi -------------------------------------------------------

  const onFileChange = (event) => {
    setFile(event.target.files?.[0] || null)
    setError('')
    setResult(null)
  }

  const onDrop = (event) => {
    event.preventDefault()
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) {
      setFile(dropped)
      setError('')
      setResult(null)
    }
  }

  const onDragOver = (event) => event.preventDefault()

  const onSubmitUrl = (event) => {
    event.preventDefault()
    setApiUrl(urlDraft.trim())
  }

  // --- Canlı waveform -----------------------------------------------------

  // AnalyserNode ile mikrofonun time-domain verisinden peak hesaplayıp
  // 60 frame'lik yığına ekliyoruz (kayıt sırasında mobilde büyüyen simge).
  const startLiveWaveform = (stream) => {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext
      if (!Ctx) return
      const ctx = new Ctx()
      audioCtxRef.current = ctx
      const source = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser

      const buffer = new Uint8Array(analyser.frequencyBinCount)
      const loop = () => {
        if (!analyserRef.current) return
        analyser.getByteTimeDomainData(buffer)
        let peak = 0
        for (let i = 0; i < buffer.length; i++) {
          const v = Math.abs((buffer[i] - 128) / 128)
          if (v > peak) peak = v
        }
        setLiveLevels((prev) => {
          const next = [...prev, Math.min(1, peak)]
          return next.length > 60 ? next.slice(-60) : next
        })
        animRef.current = requestAnimationFrame(loop)
      }
      animRef.current = requestAnimationFrame(loop)
    } catch {
      /* sessizce yoksay */
    }
  }

  const stopLiveWaveform = () => {
    if (animRef.current) cancelAnimationFrame(animRef.current)
    animRef.current = null
    analyserRef.current = null
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {})
      audioCtxRef.current = null
    }
  }

  // --- Mikrofon kaydı ----------------------------------------------------

  const startRecording = async () => {
    if (isRecording) return
    setError('')
    setMicError('')
    setResult(null)
    setLiveLevels([])

    if (
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.getUserMedia !== 'function'
    ) {
      const proto = window.location.protocol
      const host = window.location.hostname
      const msg =
        'Tarayıcı mikrofon API’sini desteklemiyor. ' +
        'Telefondan erişiyorsan bağlantının HTTPS olması gerekir ' +
        `(şu an: ${proto}//${host}). ngrok veya GitHub Pages (HTTPS) üzerinden aç.`
      setMicError(msg)
      setError(msg)
      return
    }

    if (!isSecure) {
      setMicError(
        'Bu sayfa güvenli (HTTPS) bağlamda değil. Mobil tarayıcılar mikrofonu ' +
          'sadece https:// adreslerinde açarlar. Lütfen ngrok veya GitHub Pages üzerinden gir.'
      )
      // Yine de deneyelim — bazı cihazlar izin verebilir
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      streamRef.current = stream
      startLiveWaveform(stream)

      const mimeType = pickSupportedMime()
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data)
      }
      recorder.onstop = async () => {
        stopLiveWaveform()
        const rawBlob = new Blob(chunksRef.current, {
          type: mimeType || 'audio/webm',
        })
        try {
          setConverting(true)
          const wavBlob = await blobToMonoWav(rawBlob)
          const recordedFile = new File(
            [wavBlob],
            `kayit-${Date.now()}.wav`,
            { type: 'audio/wav' }
          )
          setFile(recordedFile)
        } catch (err) {
          setError(
            'Kayıt dönüştürülemedi (' +
              (err?.message || err) +
              '). Lütfen tekrar deneyin.'
          )
        } finally {
          setConverting(false)
          if (streamRef.current) {
            streamRef.current.getTracks().forEach((t) => t.stop())
            streamRef.current = null
          }
        }
      }

      recorder.start()
      recorderRef.current = recorder
      startedAtRef.current = Date.now()
      setRecordingSeconds(0)
      setIsRecording(true)

      tickRef.current = setInterval(() => {
        const elapsed = (Date.now() - startedAtRef.current) / 1000
        setRecordingSeconds(elapsed)
      }, 200)
    } catch (err) {
      stopLiveWaveform()
      let detail = err?.message || String(err)
      if (err && err.name === 'NotAllowedError') {
        detail =
          'Mikrofon izni reddedildi. Tarayıcı ayarlarından mikrofon erişimine izin verin.'
      } else if (err && err.name === 'NotFoundError') {
        detail = 'Cihazda kullanılabilir mikrofon bulunamadı.'
      } else if (err && err.name === 'NotReadableError') {
        detail = 'Mikrofon başka bir uygulama tarafından kullanılıyor olabilir.'
      } else if (err && err.name === 'OverconstrainedError') {
        detail = 'Mikrofon, istenen özellikleri karşılamıyor.'
      } else if (err && err.name === 'SecurityError') {
        detail = 'Güvenlik nedeniyle erişim reddedildi. Sayfayı HTTPS üzerinden açın.'
      }
      const msg = 'Mikrofona erişilemedi: ' + detail
      setMicError(msg)
      setError(msg)
    }
  }

  const stopRecording = () => {
    if (!isRecording) return
    try {
      recorderRef.current?.stop()
    } catch {
      /* yok say */
    }
    if (tickRef.current) {
      clearInterval(tickRef.current)
      tickRef.current = null
    }
    setIsRecording(false)
  }

  // --- Analiz isteği -----------------------------------------------------

  const analyze = async () => {
    if (!effectiveUrl) {
      setError("Lütfen önce Backend API URL'ini girin.")
      return
    }
    if (!file) {
      setError('Lütfen bir ses dosyası seçin veya mikrofonla kayıt yapın.')
      return
    }

    setError('')
    setResult(null)
    setLoading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${effectiveUrl}/analyze`, {
        method: 'POST',
        body: formData,
        // ngrok'un yeni sürümlerinde gereken bypass header'ı
        headers: { 'ngrok-skip-browser-warning': 'true' },
      })
      if (!response.ok) {
        const detail = await safeReadDetail(response)
        throw new Error(detail || `Sunucu hatası: ${response.status}`)
      }
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err?.message || 'Bilinmeyen bir hata oluştu.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen w-full px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-4xl">
        <Header />

        <ApiUrlCard
          urlDraft={urlDraft}
          setUrlDraft={setUrlDraft}
          onSubmit={onSubmitUrl}
          health={health}
        />

        <RecorderBar
          isRecording={isRecording}
          seconds={recordingSeconds}
          onStart={startRecording}
          onStop={stopRecording}
          hasFile={!!file}
          converting={converting}
          liveLevels={liveLevels}
          isSecure={isSecure}
          micError={micError}
        />

        <UploadCard
          file={file}
          onFileChange={onFileChange}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onAnalyze={analyze}
          loading={loading}
          disabled={!effectiveUrl}
        />

        {error && (
          <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            <strong>Hata:</strong> {error}
          </div>
        )}

        {loading && <ResultsSkeleton />}

        {result && !loading && <ResultsView result={result} />}
      </div>
    </div>
  )
}

// --- Bileşenler -----------------------------------------------------------

function Header() {
  return (
    <header className="mb-8 text-center">
      <div className="inline-flex items-center gap-3 rounded-full border border-blue-100 bg-white/70 px-4 py-1.5 text-xs font-medium text-blue-700 shadow-sm">
        <span className="inline-block h-2 w-2 rounded-full bg-blue-500" />
        Stutter Detection · <a href="https://furkan.software" target="_blank" rel="noreferrer">Nafair</a> - 2026
      </div>
      <h1 className="mt-4 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
        Kekemelik Analiz Sistemi
      </h1>
      <p className="mt-2 text-sm text-slate-600 sm:text-base">
        Bir ses dosyası yükleyin ya da mikrofondan kayıt yapın; yapay zekâ
        destekli modelimiz 3 saniyelik pencereler halinde kekemelik tespiti
        yapsın.
      </p>
    </header>
  )
}

function HealthBadge({ health }) {
  if (health.state === 'idle') return null
  const map = {
    loading: {
      cls: 'border-amber-200 bg-amber-50 text-amber-800',
      icon: <Spinner small className="text-amber-600" />,
    },
    ok: {
      cls: 'border-emerald-200 bg-emerald-50 text-emerald-800',
      icon: <CheckIcon />,
    },
    warn: {
      cls: 'border-amber-200 bg-amber-50 text-amber-800',
      icon: <WarnIcon />,
    },
    error: {
      cls: 'border-red-200 bg-red-50 text-red-800',
      icon: <CrossIcon />,
    },
  }
  const tone = map[health.state] || map.error
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${tone.cls}`}
      title={health.message}
    >
      {tone.icon}
      {health.message}
    </span>
  )
}

function ApiUrlCard({ urlDraft, setUrlDraft, onSubmit, health }) {
  return (
    <form
      onSubmit={onSubmit}
      className="rounded-2xl border border-slate-200 bg-white/80 p-5 shadow-sm backdrop-blur"
    >
      <div className="flex items-center justify-between gap-2">
        <label htmlFor="api-url" className="block text-sm font-semibold text-slate-800">
          Backend API URL
        </label>
        <HealthBadge health={health} />
      </div>
      <p className="mt-1 text-xs text-slate-500">
        Ngrok veya başka bir tünel üzerinden aldığınız tam URL (ör.
        <code className="rounded bg-slate-100 px-1.5 py-0.5">
          https://abc.ngrok-free.app
        </code>
        ). Tarayıcıda saklanır.
      </p>
      <div className="mt-3 flex flex-col gap-2 sm:flex-row">
        <input
          id="api-url"
          type="url"
          placeholder="https://your-api.example.com"
          value={urlDraft}
          onChange={(e) => setUrlDraft(e.target.value)}
          className="flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
        />
        <button
          type="submit"
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300"
        >
          Kaydet
        </button>
      </div>
    </form>
  )
}

function RecorderBar({
  isRecording,
  seconds,
  onStart,
  onStop,
  hasFile,
  converting,
  liveLevels,
  isSecure,
  micError,
}) {
  // Son peak değerine göre mikrofon ikonunu büyüt (mobilde fark edilir pulse)
  const latestPeak =
    liveLevels && liveLevels.length > 0 ? liveLevels[liveLevels.length - 1] : 0
  const micScale = isRecording ? 0.85 + Math.min(1, latestPeak) * 0.5 : 1

  return (
    <section className="mt-6 rounded-2xl border border-slate-200 bg-white/80 p-4 shadow-sm backdrop-blur">
      <div className="flex flex-col items-center gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex w-full items-center gap-3 sm:w-auto">
          <div
            className={`flex h-12 w-12 items-center justify-center rounded-full text-2xl transition-transform duration-75 ${
              isRecording
                ? 'bg-red-100 text-red-700'
                : converting
                ? 'bg-indigo-100 text-indigo-700'
                : 'bg-slate-100 text-slate-500'
            }`}
            style={{ transform: `scale(${micScale})` }}
          >
            {converting ? <Spinner className="text-indigo-600" /> : '🎤'}
          </div>
          <div className="min-w-0">
            <div className="text-sm font-semibold text-slate-800">
              {converting
                ? "WAV'a dönüştürülüyor…"
                : isRecording
                ? 'Kayıt Yapılıyor…'
                : hasFile
                ? 'Kayıt Hazır'
                : 'Mikrofondan Kaydet'}
            </div>
            <div className="text-xs text-slate-500">
              {converting
                ? 'Tarayıcı 16 kHz mono PCM formatına çeviriyor'
                : isRecording
                ? `Süre: ${formatClock(seconds)}`
                : 'Tek tıkla mikrofondan kısa bir kayıt alıp analiz edin'}
            </div>
          </div>
        </div>

        <div className="flex w-full items-center sm:w-auto">
          {isRecording ? (
            <button
              type="button"
              onClick={onStop}
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-red-600 px-6 py-3 text-base font-semibold text-white shadow-sm transition hover:bg-red-700 sm:w-auto sm:px-5 sm:py-2 sm:text-sm"
            >
              <span className="inline-block h-3 w-3 rounded-sm bg-white" />
              Dur
            </button>
          ) : (
            <button
              type="button"
              onClick={onStart}
              disabled={converting}
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-slate-900 px-6 py-3 text-base font-semibold text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-500 sm:w-auto sm:px-5 sm:py-2 sm:text-sm"
            >
              <span className="inline-block h-3 w-3 rounded-full bg-red-500" />
              Mikrofondan Kaydet
            </button>
          )}
        </div>
      </div>

      {/* Mobil/HTTP bağlantılarında mikrofon erişim uyarısı */}
      {!isSecure && !isRecording && !converting && (
        <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          ⚠️ Sayfa güvenli (HTTPS) bağlamda değil. Telefondan mikrofona erişmek
          için <code className="font-mono">https://</code> üzerinden açın
          (ngrok veya GitHub Pages).
        </div>
      )}

      {/* Dalga formu sadece masaüstünde */}
      {(isRecording || converting) && (
        <div className="mt-3 hidden sm:block">
          <LiveWaveform levels={liveLevels} active={isRecording} />
        </div>
      )}

      {micError && !isRecording && (
        <div className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
          {micError}
        </div>
      )}
    </section>
  )
}

function LiveWaveform({ levels, active }) {
  const bars = levels && levels.length > 0 ? levels : []
  const placeholders = Array.from({ length: 60 }, () => 0)

  return (
    <div className="relative h-12 w-full overflow-hidden rounded-md bg-slate-50">
      <div className="flex h-full w-full items-center justify-between gap-[2px] px-0.5">
        {(active ? bars : placeholders).map((v, i) => {
          const heightPct = active ? Math.max(6, Math.min(100, v * 100)) : 6
          return (
            <div
              key={i}
              className={`flex-1 rounded-sm transition-all duration-75 ${
                active ? 'bg-red-500/80' : 'bg-slate-300'
              }`}
              style={{ height: `${heightPct}%`, minHeight: '2px' }}
            />
          )
        })}
      </div>
      {active && bars.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-[10px] font-mono text-slate-400">
          Ses bekleniyor…
        </div>
      )}
    </div>
  )
}

function UploadCard({
  file,
  onFileChange,
  onDrop,
  onDragOver,
  onAnalyze,
  loading,
  disabled,
}) {
  return (
    <section className="mt-6 rounded-2xl border border-slate-200 bg-white/80 p-5 shadow-sm backdrop-blur">
      <h2 className="text-sm font-semibold text-slate-800">Ses Dosyası</h2>

      <label
        onDrop={onDrop}
        onDragOver={onDragOver}
        htmlFor="audio-file"
        className="mt-3 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 px-4 py-8 text-center text-sm text-slate-600 transition hover:border-blue-400 hover:bg-blue-50"
      >
        <span className="text-2xl">📁</span>
        <span className="font-medium text-slate-700">
          {file ? file.name : 'Bir .wav dosyası seçin ya da buraya sürükleyin'}
        </span>
        <span className="text-xs text-slate-500">
          Tek dosya · ses kaydı yaptıysanız burada görünür
        </span>
        <input
          id="audio-file"
          type="file"
          accept="audio/*,.wav,.webm,.ogg,.m4a"
          onChange={onFileChange}
          className="hidden"
        />
      </label>

      <div className="mt-4 flex flex-col-reverse items-stretch gap-2 sm:flex-row sm:items-center sm:justify-end sm:gap-3">
        {file && !loading && (
          <button
            type="button"
            onClick={() => {
              setFile(null)
            }}
            className="inline-flex w-full items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 sm:w-auto sm:border-0 sm:bg-transparent sm:px-0 sm:py-0 sm:shadow-none sm:hover:bg-transparent sm:hover:underline sm:hover:underline-offset-2"
          >
            Seçimi temizle
          </button>
        )}
        <button
          type="button"
          onClick={onAnalyze}
          disabled={loading || !file || disabled}
          className={`inline-flex w-full items-center justify-center gap-2 rounded-lg px-4 py-3 text-base font-semibold text-white shadow-sm transition disabled:cursor-not-allowed sm:w-auto sm:min-w-[140px] sm:px-4 sm:py-2 sm:text-sm ${
            loading
              ? 'bg-indigo-500'
              : 'bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300'
          }`}
        >
          {loading ? (
            <>
              <Spinner />
              Yükleniyor…
            </>
          ) : (
            <>Analiz Et</>
          )}
        </button>
      </div>
    </section>
  )
}

function Spinner({ small = false, className = 'text-white' }) {
  const sizeCls = small ? 'h-3.5 w-3.5' : 'h-4 w-4'
  return (
    <svg
      className={`${sizeCls} animate-spin ${className}`}
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
        opacity="0.25"
      />
      <path
        d="M22 12a10 10 0 0 1-10 10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg
      viewBox="0 0 20 20"
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M4 11l4 4L16 5" />
    </svg>
  )
}

function CrossIcon() {
  return (
    <svg
      viewBox="0 0 20 20"
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M5 5l10 10M15 5L5 15" />
    </svg>
  )
}

function WarnIcon() {
  return (
    <svg
      viewBox="0 0 20 20"
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M10 6v4" />
      <circle cx="10" cy="14" r="0.5" fill="currentColor" />
    </svg>
  )
}

function ResultsSkeleton() {
  return (
    <section className="mt-6 space-y-5">
      <div className="h-5 w-32 animate-pulse rounded bg-slate-200" />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="h-24 animate-pulse rounded-2xl bg-slate-100" />
        <div className="h-24 animate-pulse rounded-2xl bg-slate-100" />
      </div>
      <div className="h-10 animate-pulse rounded-xl bg-slate-100" />
      <div className="space-y-2">
        <div className="h-10 animate-pulse rounded-xl bg-slate-100" />
        <div className="h-10 animate-pulse rounded-xl bg-slate-100" />
        <div className="h-10 animate-pulse rounded-xl bg-slate-100" />
      </div>
    </section>
  )
}

function ResultsView({ result }) {
  const { total_duration, stutter_count, chunks } = result
  return (
    <section className="mt-6 space-y-5">
      <h2 className="text-base font-semibold text-slate-800">Sonuçlar</h2>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <MetricCard
          label="Toplam Süre"
          value={`${total_duration ?? 0} sn`}
          tone="blue"
        />
        <MetricCard
          label="Tespit Edilen Kekemelik"
          value={stutter_count ?? 0}
          tone={stutter_count > 0 ? 'red' : 'green'}
        />
      </div>

      <Timeline
        totalDuration={total_duration}
        chunks={chunks || []}
        waveform={result.waveform || []}
      />

      <div className="rounded-2xl border border-slate-200 bg-white/80 p-5 shadow-sm backdrop-blur">
        <h3 className="text-sm font-semibold text-slate-800">Parça Listesi</h3>
        {(!chunks || chunks.length === 0) && (
          <p className="mt-2 text-sm text-slate-500">
            Analiz edilecek parça bulunamadı.
          </p>
        )}
        <ol className="mt-3 space-y-2">
          {(chunks || []).map((chunk, idx) => (
            <TimelineRow key={`${chunk.start_time}-${idx}`} chunk={chunk} />
          ))}
        </ol>
      </div>
    </section>
  )
}

function MetricCard({ label, value, tone }) {
  const tones = {
    blue: 'border-blue-200 bg-blue-50 text-blue-900',
    red: 'border-red-200 bg-red-50 text-red-900',
    green: 'border-emerald-200 bg-emerald-50 text-emerald-900',
  }
  return (
    <div
      className={`rounded-2xl border p-5 shadow-sm ${tones[tone] || tones.blue}`}
    >
      <div className="text-xs font-semibold uppercase tracking-wide opacity-80">
        {label}
      </div>
      <div className="mt-2 text-3xl font-bold">{value}</div>
    </div>
  )
}

function Timeline({ totalDuration, chunks, waveform }) {
  const safeDuration = totalDuration && totalDuration > 0 ? totalDuration : 0
  const segments = (chunks || []).map((c) => {
    const start = Number(c.start_time) || 0
    const end = Number(c.end_time) || start
    const dur = Math.max(0.001, end - start)
    return { ...c, start, end, dur }
  })
  const totalSeg = segments.reduce((acc, s) => acc + s.dur, 0) || 1

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/80 p-5 shadow-sm backdrop-blur">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-800">Yatay Zaman Çizelgesi</h3>
        <span className="text-xs font-mono text-slate-500">
          {safeDuration.toFixed ? safeDuration.toFixed(2) : safeDuration} sn
        </span>
      </div>

      <div className="relative">
        {/* Üstte: chunk label'larına göre renklenen segment bar */}
        <div className="h-8 w-full overflow-hidden rounded-lg bg-slate-100">
          <div className="flex h-full w-full">
            {segments.length === 0 && (
              <div className="h-full w-full animate-pulse bg-slate-200" />
            )}
            {segments.map((seg, idx) => {
              const widthPct = (seg.dur / totalSeg) * 100
              const isStutter = seg.label === 'KEKEMELİK'
              return (
                <div
                  key={`${seg.start}-${idx}`}
                  className={`group relative flex h-full items-center justify-center text-[10px] font-semibold text-white transition ${
                    isStutter
                      ? 'bg-red-500/90 hover:bg-red-600'
                      : 'bg-emerald-500/90 hover:bg-emerald-600'
                  }`}
                  style={{ width: `${widthPct}%` }}
                  title={`${seg.start}sn – ${seg.end}sn · ${seg.label} · %${(
                    (seg.confidence || 0) * 100
                  ).toFixed(1)}`}
                >
                  <span className="hidden truncate px-1 sm:inline">
                    {seg.label === 'KEKEMELİK' ? '🔴' : '🟢'}
                  </span>
                  <div className="pointer-events-none absolute -top-9 left-1/2 z-10 hidden -translate-x-1/2 whitespace-nowrap rounded-md bg-slate-900 px-2 py-1 text-[11px] font-medium text-white shadow-lg group-hover:block">
                    {seg.label} · %{((seg.confidence || 0) * 100).toFixed(1)}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Altta: ses dalgası (waveform) — renkler segment'lere göre */}
        <div className="mt-2">
          <WaveformBar
            values={waveform || []}
            segments={segments}
            totalDuration={safeDuration}
          />
        </div>

        <div className="mt-1 flex justify-between text-[10px] font-mono text-slate-400">
          <span>0.0 sn</span>
          <span>{(safeDuration / 2).toFixed(1)} sn</span>
          <span>{safeDuration.toFixed(1)} sn</span>
        </div>
      </div>

      <div className="mt-3 flex items-center gap-3 text-xs text-slate-500">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded bg-emerald-500" />
          Akıcı
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded bg-red-500" />
          Kekemelik
        </span>
      </div>
    </div>
  )
}

// Ses dalgası: 60 çubukluk peak değerlerini chunk label'larına göre renklendirir;
// segment sınırları dikey ince çizgiyle gösterilir.
function WaveformBar({ values, segments, totalDuration }) {
  if (!values || values.length === 0) {
    return (
      <div className="flex h-12 w-full items-center justify-center rounded-md bg-slate-50 text-[10px] font-mono text-slate-400">
        Waveform verisi yok
      </div>
    )
  }

  const N = values.length
  const safeDur = totalDuration > 0 ? totalDuration : 1
  const colorForIndex = (idx) => {
    if (!segments || segments.length === 0) return 'bg-slate-300'
    const t = (idx / N) * safeDur
    for (const seg of segments) {
      if (t >= seg.start && t < seg.end) {
        return seg.label === 'KEKEMELİK' ? 'bg-red-400/80' : 'bg-emerald-400/80'
      }
    }
    return 'bg-slate-300'
  }
  const boundaryPcts = (segments || [])
    .slice(1)
    .map((s) => (s.start / safeDur) * 100)
    .filter((p) => p > 0 && p < 100)

  return (
    <div className="relative h-12 w-full overflow-hidden rounded-md bg-slate-50">
      <div className="flex h-full w-full items-center justify-between gap-[2px] px-0.5">
        {values.map((v, i) => {
          const heightPct = Math.max(6, Math.min(100, v * 100))
          return (
            <div
              key={i}
              className={`flex-1 rounded-sm ${colorForIndex(i)} transition`}
              style={{ height: `${heightPct}%`, minHeight: '2px', opacity: 0.85 }}
            />
          )
        })}
      </div>
      {boundaryPcts.map((pct, i) => (
        <div
          key={`bnd-${i}`}
          className="pointer-events-none absolute top-0 h-full w-px bg-slate-900/40"
          style={{ left: `${pct}%` }}
        />
      ))}
    </div>
  )
}

function TimelineRow({ chunk }) {
  const isStutter = chunk.label === 'KEKEMELİK'
  return (
    <li
      className={`flex items-center justify-between gap-3 rounded-xl border px-3 py-2 text-sm transition ${
        isStutter
          ? 'border-red-200 bg-red-50 text-red-900'
          : 'border-emerald-200 bg-emerald-50 text-emerald-900'
      }`}
    >
      <div className="flex items-center gap-3">
        <span
          className={`inline-flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold ${
            isStutter ? 'bg-red-200 text-red-900' : 'bg-emerald-200 text-emerald-900'
          }`}
        >
          {isStutter ? '🔴' : '🟢'}
        </span>
        <span className="font-mono">
          {chunk.start_time}sn – {chunk.end_time}sn
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span className="font-semibold">{chunk.label}</span>
        <span className="rounded-md bg-white/70 px-2 py-0.5 text-xs font-mono">
          {(chunk.confidence * 100).toFixed(1)}%
        </span>
      </div>
    </li>
  )
}

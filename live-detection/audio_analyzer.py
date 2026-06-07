import pyaudio
import wave
import librosa
import numpy as np
import pandas as pd
import joblib
import os
import time
import warnings

# Librosa uyarılarını gizle
warnings.filterwarnings('ignore')

class AudioRecorder:
    def __init__(self, rate=16000, chunk=1024, channels=1):
        self.rate = rate
        self.chunk = chunk
        self.channels = channels
        self.format = pyaudio.paInt16

    def record_audio(self, duration=60, filename="session.wav"):
        """
        Belirtilen süre boyunca mikrofondan ses kaydeder ve .wav dosyası olarak kaydeder.
        """
        p = pyaudio.PyAudio()

        print(f"\n🎙️  Kayıt Başlıyor... ({duration} saniye)")
        print("   Lütfen konuşmaya başlayın...")

        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)

        frames = []

        # Kayıt döngüsü
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
            
            # Geri sayım göstergesi (her saniye)
            if i % int(self.rate / self.chunk) == 0:
                remaining = duration - int(i / (self.rate / self.chunk))
                print(f"   Kalan Süre: {remaining} sn ", end='\r')

        print("\n✅ Kayıt Tamamlandı.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Dosyayı kaydet
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"   Dosya kaydedildi: {filename}")
        return filename

class StutterDetector:
    def __init__(self, model_path='stutter_rf_model.pkl', scaler_path='scaler.pkl', threshold=0.625):
        """
        Eğitilmiş modeli ve scaler'ı yükler.
        """
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model veya Scaler dosyası bulunamadı! Önce 'train_model.py' çalıştırın.")
            
        print("\n🧠 Model ve Scaler Yükleniyor...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.sample_rate = 16000 # Model eğitimi ve analiz için standart SR (SEP-28k)
        # Kekemelik kararı için eşik değeri (Fluent F1'i maksimize eden değer)
        self.threshold = threshold
        print(f"   Model yüklendi (threshold={self.threshold}).")

    def extract_features(self, y, sr):
        """
        Ses verisinden MFCC özelliklerini çıkarır.
        Eğitim verisindeki (13,) formatına uygun ortalama değerleri döndürür.
        """
        # 13 MFCC katsayısı çıkar
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Zaman ekseni boyunca ortalamasını al (13, T) -> (13,)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean

    def analyze_file(self, file_path, chunk_duration=3):
        """
        Ses dosyasını parçalara böler ve her parça için kekemelik tahmini yapar.
        """
        print(f"\n🔍 Analiz Ediliyor: {file_path}")
        
        # Dosyayı yükle
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        total_duration = librosa.get_duration(y=y, sr=sr)
        print(f"   Toplam Süre: {total_duration:.2f} saniye")
        
        results = []
        samples_per_chunk = int(chunk_duration * sr)
        
        # Parça parça işle
        for i in range(0, len(y), samples_per_chunk):
            chunk = y[i:i+samples_per_chunk]
            
            # Çok kısa parçaları atla (< 1 saniye)
            if len(chunk) < sr:
                continue
            
            # Sessizlik kontrolü (Noise Gate)
            rms = np.sqrt(np.mean(chunk**2))
            if rms < 0.005: # Eşik değeri
                start_time = i / sr
                end_time = (i + len(chunk)) / sr
                results.append({
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "is_stutter": 0,
                    "confidence": 1.0,
                    "label": "SESSİZ/AKICI"
                })
                continue
                
            # Özellik Çıkarımı
            features = self.extract_features(chunk, sr)
            
            # Ölçeklendirme (Scaler)
            # Reshape (1, -1) çünkü tek bir örnek tahmin ediyoruz
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Tahmin
            # Tahmin ve Eşik (Threshold) Ayarı
            probs = self.model.predict_proba(features_scaled)[0]
            
            # probs[1], sesin Kekemelik (1) olma olasılığıdır.
            # Eşik 0.625 (Fluent F1'i maksimize eden değer)
            stutter_threshold = self.threshold
            
            if probs[1] > stutter_threshold:
                prediction = 1
                confidence = probs[1]
            else:
                prediction = 0
                confidence = probs[0]
            
            # Zaman damgaları
            start_time = i / sr
            end_time = (i + len(chunk)) / sr
            
            results.append({
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "is_stutter": int(prediction),
                "confidence": round(confidence, 2),
                "label": "KEKEMELİK" if prediction == 1 else "AKICI"
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # 1. Kayıt Ayarları
    DURATION = 10 # Test için 10 saniye (Gerçek kullanımda 60 yapabilirsiniz)
    FILENAME = "test_session.wav"
    
    try:
        # 2. Ses Kaydı
        recorder = AudioRecorder()
        recorder.record_audio(duration=DURATION, filename=FILENAME)
        
        # 3. Analiz
        detector = StutterDetector()
        df_results = detector.analyze_file(FILENAME, chunk_duration=3)
        
        # 4. Sonuçları Göster
        print("\n📊 ANALİZ SONUÇLARI (Zaman Çizelgesi)")
        print("="*60)
        print(f"{'Zaman Aralığı':<20} | {'Durum':<15} | {'Güven':<10}")
        print("-" * 60)
        
        stutter_count = 0
        for index, row in df_results.iterrows():
            time_str = f"{row['start_time']}s - {row['end_time']}s"
            label = row['label']
            conf = f"%{row['confidence']*100:.1f}"
            
            # Kekemelik varsa kırmızı (veya belirgin) yazdırılabilir, burada düz metin
            prefix = "🔴 " if row['is_stutter'] == 1 else "🟢 "
            
            print(f"{prefix} {time_str:<17} | {label:<15} | {conf:<10}")
            
            if row['is_stutter'] == 1:
                stutter_count += 1
                
        print("="*60)
        print(f"Toplam Parça: {len(df_results)}")
        print(f"Tespit Edilen Kekemelik Sayısı: {stutter_count}")
        
        # İsterseniz CSV olarak kaydedebilirsiniz
        # df_results.to_csv("analiz_sonuclari.csv", index=False)
        
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")
        print("Gerekli kütüphanelerin yüklü olduğundan emin olun:")
        print("pip install pyaudio librosa joblib pandas scikit-learn")

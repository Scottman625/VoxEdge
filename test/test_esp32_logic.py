import argparse
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from collections import deque

# --- 參數設定 ---
MODEL_PATH = "kws_model_esp32.h5"
LABELS = ["kai", "qi", "guan", "bi", "silence"]

SAMPLE_RATE = 16000
DURATION = 0.512
WINDOW_SIZE = int(SAMPLE_RATE * DURATION)

# ⚖️ 你的黃金參數
TARGET_RMS = 0.25    
MAX_GAIN = 25.0      

MIN_RMS_THRESHOLD = 0.01  # 你可以實測調整，大約是 -40 ~ -35 dB 左右

# 滑動視窗設定
# 每次移動多少才做一次預測？
# 0.16s (160ms) 大約是 1/3 個視窗。這樣 "Kai Qi" 這種快速詞組也能被切開。
INFERENCE_STRIDE_SECONDS = 0.16 
STRIDE_SIZE = int(SAMPLE_RATE * INFERENCE_STRIDE_SECONDS)

CONFIDENCE_THRESHOLD = 0.75 # 稍微拉高，因為我們預測頻率變高了，要過濾雜訊

N_FFT_BINS = 64

# 狀態變數
audio_buffer = deque(maxlen=WINDOW_SIZE) # 永遠只裝 0.512s
samples_since_last_inference = 0
last_label = "silence"
last_detection_time = 0

print(f"⏳ 載入模型: {MODEL_PATH} ...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    exit()

print(f"=== 滑動視窗模式 (Sliding Window) ===")
print(f"📉 EQ: Bin0 x0.5 | Bin1 x0.75")
print(f"🌊 預測頻率: 每 {INFERENCE_STRIDE_SECONDS}s 一次")

def user_tuned_feature_extraction(raw_audio):
    # 這裡保持你調好的邏輯不變
    
    # Gain Control
    rms = np.sqrt(np.mean(raw_audio**2))
    if rms < 1e-8: rms = 1e-8
    
    if rms < 0.005:
        gain = 1.0
    else:
        gain = TARGET_RMS / rms
    
    if gain > MAX_GAIN: gain = MAX_GAIN
    
    processed_audio = raw_audio * gain
    processed_audio = np.clip(processed_audio, -1.0, 1.0)

    # FFT
    window = np.hanning(len(processed_audio))
    spectrum = np.fft.rfft(window * processed_audio)
    mag = np.abs(spectrum)

    # Binning
    chunk_size = len(mag) // N_FFT_BINS
    features = np.zeros(N_FFT_BINS)

    for i in range(N_FFT_BINS):
        start = i * chunk_size
        end = start + chunk_size
        if end > len(mag): end = len(mag)
        if start < end:
            features[i] = np.mean(mag[start:end])

    # 🛑 你的 EQ 設定
    features[0] = features[0] * 0.5
    features[1] = features[1] * 0.75
    
    # Log
    features = np.log1p(features)
    return features

def process_audio_chunk(indata):
    global audio_buffer, samples_since_last_inference, last_label, last_detection_time

    # 1. 把新數據塞進 buffer (舊的會自動被擠出去)
    audio_buffer.extend(indata)
    samples_since_last_inference += len(indata)

    # 2. 檢查是否該做預測了 (Stride Check)
    # 只有當 buffer 滿了，且距離上次預測已經過了 STRIDE_SIZE
    if len(audio_buffer) == WINDOW_SIZE and samples_since_last_inference >= STRIDE_SIZE:
        
        # 歸零計數器
        samples_since_last_inference = 0
        
        # 提取特徵
        full_audio = np.array(audio_buffer)

        # ✅ 先算原始 RMS（還沒 AGC）
        rms = np.sqrt(np.mean(full_audio**2))
        if rms < MIN_RMS_THRESHOLD:
            # 太小聲，當作背景音 / 靜音，直接略過這次推論
            return
        features = user_tuned_feature_extraction(full_audio)

        plot_all(full_audio, title="說 bi 的一段")
        
        # 預測
        input_tensor = features.reshape(1, -1)
        prediction = model.predict(input_tensor, verbose=0)[0]
        label_idx = np.argmax(prediction)
        score = prediction[label_idx]
        current_label = LABELS[label_idx]
        
        now = time.time()

        # 3. 智慧去重邏輯 (Smart Debounce)
        if current_label != "silence" and score > CONFIDENCE_THRESHOLD:
            
            # 情況 A: 這是新指令 (例如上次是 Kai，這次是 Qi) -> 馬上觸發！
            if current_label != last_label:
                print(f">>> 🎉 識別指令: 【 {current_label} 】 ({score:.2f}) <<<")
                last_label = current_label
                last_detection_time = now
            
            # 情況 B: 這是舊指令 (例如上次是 Kai，這次還是 Kai)
            else:
                # 只有過了 0.8秒 才能再次觸發同一個字 (防止重複觸發)
                if (now - last_detection_time) > 0.8:
                    print(f">>> 🎉 識別指令: 【 {current_label} 】 ({score:.2f}) (重複)")
                    last_detection_time = now
        
        # 如果是 silence，且持續了一段時間，重置 last_label
        elif current_label == "silence":
             if (now - last_detection_time) > 0.5:
                 last_label = "silence"

def audio_callback(indata, frames, time, status):
    if status: print(status)
    process_audio_chunk(indata.flatten())

def user_tuned_feature_extraction_debug(raw_audio):
    # ---- 跟你原本的一樣，多回傳中間結果 ----
    rms = np.sqrt(np.mean(raw_audio**2))
    if rms < 1e-8: 
        rms = 1e-8
    
    if rms < 0.005:
        gain = 1.0
    else:
        gain = TARGET_RMS / rms
    
    if gain > MAX_GAIN: 
        gain = MAX_GAIN
    
    processed_audio = raw_audio * gain
    processed_audio = np.clip(processed_audio, -1.0, 1.0)

    window = np.hanning(len(processed_audio))
    windowed = window * processed_audio
    spectrum = np.fft.rfft(windowed)
    mag = np.abs(spectrum)

    chunk_size = len(mag) // N_FFT_BINS
    features = np.zeros(N_FFT_BINS)

    for i in range(N_FFT_BINS):
        start = i * chunk_size
        end = start + chunk_size
        if end > len(mag): 
            end = len(mag)
        if start < end:
            features[i] = np.mean(mag[start:end])

    features[0] *= 0.5
    features[1] *= 0.75
    
    features = np.log1p(features)

    return processed_audio, windowed, mag, features, gain, rms

def plot_all(raw_audio, title=""):
    processed_audio, windowed, mag, features, gain, rms = user_tuned_feature_extraction_debug(raw_audio)

    t = np.arange(len(raw_audio)) / SAMPLE_RATE
    freqs = np.fft.rfftfreq(len(windowed), d=1.0/SAMPLE_RATE)

    plt.figure(figsize=(12, 10))
    
    # 1. 原始波形
    plt.subplot(4, 1, 1)
    plt.plot(t, raw_audio)
    plt.title(f"{title} - Raw waveform (rms={rms:.4f})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # 2. AGC + clip 後波形
    plt.subplot(4, 1, 2)
    plt.plot(t, processed_audio)
    plt.title(f"Processed waveform (gain={gain:.2f})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 3. 頻譜 (magnitude)
    plt.subplot(4, 1, 3)
    plt.semilogy(freqs, mag + 1e-8)  # 用 log 軸比較好看
    plt.title("Magnitude Spectrum (rfft)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X(f)|")

    # 4. 64 維 features
    plt.subplot(4, 1, 4)
    plt.bar(np.arange(len(features)), features)
    plt.title("64-bin log features")
    plt.xlabel("Bin index")
    plt.ylabel("log(1 + mean_mag)")

    plt.tight_layout()
    plt.show()

def main():
    # 為了讓滑動更順暢，block size 設小一點
    block_size = int(SAMPLE_RATE * 0.02) # 20ms per block
    try:
        with sd.InputStream(callback=audio_callback, channels=1, 
                            samplerate=SAMPLE_RATE, blocksize=block_size):
            print("🎧 滑動視窗模式啟動!")
            print("請測試連讀詞組：'開啟' (Kai Qi) 或 '關閉' (Guan Bi)...")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 停止")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    main()

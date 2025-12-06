import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --- 參數設定 ---
SAMPLE_RATE = 16000
DURATION = 0.256 
DATASET_PATH = "dataset_synthetic_expanded_light"
# DATASET_PATH = "dataset"   # 或 "dataset" 根據你要用合成還是錄音資料
# LABELS = ["開", "啟", "打" ,"關", "閉", "掉", "silence"]
LABELS = ["開","啟","關","閉", "silence"]
feature_dim = 0 # 自動計算用

# FFT-based feature options
N_FFT_BINS = 64  # number of low-frequency FFT bins to keep (32 or 64)

# RMS normalization options (kept for compatibility)
NORMALIZE_RMS = True
TARGET_RMS = 0.10
MAX_GAIN = 10.0

# --- 前處理選項 ---
# 是否對每個音檔做 RMS 正規化 (將音量調整到目標 RMS)
NORMALIZE_RMS = True
# 目標 RMS（float，範圍 0..1），越大音量越大。根據上面 check_wavs 的觀察，gTTS 的 RMS 約 0.09-0.13
# 我們可以把 edge-tts 放大到類似範圍，例如 0.10
TARGET_RMS = 0.10
# 防止過度放大噪音，限制最大增益倍數
MAX_GAIN = 10.0

def add_noise(audio_data, noise_factor=0.005):
    """
    在原始音訊中加入隨機白噪音
    noise_factor: 噪音強度
    """
    noise = np.random.randn(len(audio_data))
    augmented_data = audio_data + noise_factor * noise
    return augmented_data

# # --- 1. 特徵提取函數 (模擬 FPGA 的前處理) ---
# def extract_features(file_path):
#     # 載入音訊
#     audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    
#     # 補零或裁切確保長度一致
#     if len(audio) < SAMPLE_RATE * DURATION:
#         audio = np.pad(audio, (0, int(SAMPLE_RATE * DURATION) - len(audio)))
#     else:
#         audio = audio[:int(SAMPLE_RATE * DURATION)]

#     # 可選：RMS 正規化，將較小能量的語音放大到 TARGET_RMS
#     if NORMALIZE_RMS:
#         eps = 1e-8
#         cur_rms = np.sqrt(np.mean(audio**2)) if audio.size > 0 else 0.0
#         if cur_rms > eps:
#             gain = TARGET_RMS / cur_rms
#             if gain > MAX_GAIN:
#                 gain = MAX_GAIN
#             if gain != 1.0:
#                 audio = audio * gain
#         # 限幅在 -1..1 範圍以避免後續轉換溢位
#         audio = np.clip(audio, -1.0, 1.0)

#     # FFT-based feature: window the signal and compute full-window FFT magnitude,
#     # keep only the first N_FFT_BINS (low-frequency magnitudes)
#     window = np.hanning(len(audio))
#     spectrum = np.fft.rfft(window * audio)
#     mag = np.abs(spectrum)
#     # take first N_FFT_BINS (or pad if too short)
#     if len(mag) < N_FFT_BINS:
#         mag = np.pad(mag, (0, N_FFT_BINS - len(mag)))
#     else:
#         mag = mag[:N_FFT_BINS]

#     # optional log scaling to compress dynamic range
#     feat = np.log1p(mag)
#     return feat.flatten()

# --- 1. 特徵提取函數 (完全模擬 ESP32 邏輯) ---
def extract_features(file_path):
    # 先載入完整的音檔 (不限制 duration)
    audio_full, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    
    target_len = int(SAMPLE_RATE * DURATION) # 4096 點

    # ⚠️ 修改 2: 聰明的裁切 (Auto-Center)
    # 因為 0.25秒 很短，我們不能只取開頭，要取「最大聲」的那一段
    if len(audio_full) > target_len:
        # 找出能量最強的地方 (RMS 最大處)
        frame_len = 512
        hop_len = 256
        rms = librosa.feature.rms(y=audio_full, frame_length=frame_len, hop_length=hop_len)[0]
        max_rms_idx = np.argmax(rms)
        # 換算回時間點
        center_sample = max_rms_idx * hop_len + frame_len // 2
        
        start = max(0, center_sample - target_len // 2)
        end = start + target_len
        
        # 邊界檢查
        if end > len(audio_full):
            end = len(audio_full)
            start = max(0, end - target_len)
            
        audio = audio_full[start:end]
    else:
        # 如果音檔太短，就補零
        audio = np.pad(audio_full, (0, target_len - len(audio_full)))

    # --- RMS 正規化 (保持不變) ---
    if NORMALIZE_RMS:
        eps = 1e-8
        cur_rms = np.sqrt(np.mean(audio**2)) if audio.size > 0 else 0.0
        if cur_rms > eps:
            gain = TARGET_RMS / cur_rms
            if gain > MAX_GAIN:
                gain = MAX_GAIN
            if gain != 1.0:
                audio = audio * gain
        audio = np.clip(audio, -1.0, 1.0)

    # --- FFT 處理 (模擬 ESP32) ---
    window = np.hanning(len(audio))
    spectrum = np.fft.rfft(window * audio)
    mag = np.abs(spectrum)
    
    # ⚠️ 修改 3: 頻率均勻採樣 (Frequency Downsampling)
    # 這一步是關鍵！讓 Python 也能看到 8000Hz 的高頻
    # 原本是 mag[:N_FFT_BINS]，那是錯的，只能看到低音
    
    step = len(mag) // N_FFT_BINS
    if step < 1: step = 1
    
    # 建立索引：0, step, 2*step ...
    indices = np.arange(0, N_FFT_BINS) * step
    
    # 防止索引越界
    indices = np.clip(indices, 0, len(mag) - 1)
    
    # 取值
    downsampled_mag = mag[indices]

    # Log scaling
    feat = np.log1p(downsampled_mag)
    return feat.flatten()

# --- 2. 準備資料 ---
X = []
y = []

print("🔄 處理資料中...")
for i, label in enumerate(LABELS):
    label_dir = os.path.join(DATASET_PATH, label)
    for file in os.listdir(label_dir):
        if file.endswith(".wav"):
            path = os.path.join(label_dir, file)
            features = extract_features(path)
            X.append(features)
            y.append(i)

X = np.array(X)
y = np.array(y)
feature_dim = X.shape[1]

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. 建立模型 (硬體友善版) ---
# 我們只用 Dense 層，因為這在 Verilog 裡就是單純的矩陣乘法
model = models.Sequential([
    layers.Input(shape=(feature_dim,)),
    layers.Dense(32, activation='relu'), # 第一層隱藏層 (Verilog: 32組 MAC)
    layers.Dense(32, activation='relu'), # 第二層隱藏層 (Verilog: 32組 MAC)
    layers.Dense(len(LABELS), activation='softmax') # 輸出層
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- 4. 訓練 ---
print("🚀 開始訓練...")
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

# --- 5. 儲存模型 ---
model.save("kws_model.h5")
print("💾 模型已儲存為 kws_model.h5")

# --- 6. (進階) 匯出權重給 FPGA 參考 ---
# 這裡示範如何看第一層的權重，這就是你要燒進 ROM 的東西
weights, biases = model.layers[0].get_weights()
print(f"\n🔍 硬體預覽：第一層權重形狀 {weights.shape} (Input x Neurons)")
print("這些數字將來會被量化成 8-bit 整數存入 Block RAM。")

import argparse
import glob
import os
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time
import collections

# --- 1. 參數設定 ---
MODEL_PATH = "kws_model.h5"
LABELS = ["kai", "qi", "guan", "bi", "silence"]
RUNTIME_LABELS = LABELS.copy()

# VAD 設定
VAD_THRESHOLD = 0.01     # 觸發門檻 (建議比 Silence 高一點)
SAMPLE_RATE = 16000
DURATION = 0.512         # 必須與訓練一致
WINDOW_SIZE = int(SAMPLE_RATE * DURATION) 

# 特徵提取設定
N_FFT_BINS = 64
NORMALIZE_RMS = True
TARGET_RMS = 0.10
MAX_GAIN = 20.0          # 稍微放寬 Gain 上限
RMS_EPS = 1e-8

# 緩衝區設定
BLOCK_DURATION = 0.05    # 縮短 callback 間隔以獲得更快的反應 (50ms)
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

CONFIDENCE_THRESHOLD = 0.85 
COOLDOWN_TIME = 1.0      # 辨識後休息 1 秒

# --- 狀態機變數 ---
STATE_IDLE = 0
STATE_RECORDING = 1
STATE_COOLDOWN = 2

current_state = STATE_IDLE
recording_buffer = []    # 用來存觸發後的聲音
state_timer = 0          # 用來計算 Cooldown

# --- 載入模型 ---
model = None
input_shape = None

print(f"=== 智慧語音偵測 (ESP32 模擬模式) ===")
print(f"🎤 麥克風區塊: {BLOCK_DURATION*1000:.0f}ms")
print(f"🧠 模型視窗: {DURATION}s ({WINDOW_SIZE} samples)")
print(f"⚡ VAD 門檻: {VAD_THRESHOLD}")

def predict_from_audio(full_audio: np.ndarray):
    """
    進行特徵提取與預測 (與訓練邏輯完全一致)
    """
    # 1. 長度確保 (Padding / Trimming)
    if len(full_audio) < WINDOW_SIZE:
        pad = np.zeros(WINDOW_SIZE - len(full_audio), dtype=full_audio.dtype)
        full_audio = np.concatenate([full_audio, pad]) # 聲音在左，補零在右
    else:
        full_audio = full_audio[:WINDOW_SIZE] # 切掉多餘的

    # 2. RMS 正規化 (關鍵！)
    if NORMALIZE_RMS:
        cur_rms = np.sqrt(np.mean(full_audio**2)) if full_audio.size > 0 else 0.0
        
        # 如果聲音太小，視為 Silence，不放大 (避免放大雜訊)
        if cur_rms < 0.002: 
            gain = 1.0 
        else:
            gain = TARGET_RMS / cur_rms
            if gain > MAX_GAIN: gain = MAX_GAIN
            
        full_audio = full_audio * gain
        full_audio = np.clip(full_audio, -1.0, 1.0)

    # 3. 特徵提取 (FFT -> Log)
    window = np.hanning(len(full_audio))
    spectrum = np.fft.rfft(window * full_audio)
    mag = np.abs(spectrum)
    
    # 確保 Bin 數量正確
    if len(mag) < N_FFT_BINS:
        mag = np.pad(mag, (0, N_FFT_BINS - len(mag)))
    else:
        # 簡單的 Binning (取前 N 個低頻)
        # 注意：如果訓練時用了平均法 (chunk mean)，這裡也要用一樣的
        # 假設訓練是用簡單的 slice (因為 extract_features_from_wave 範例看起來是 slice 或 chunk)
        # 為了保險，這裡用與 train_finetune.py 相同的邏輯:
        chunk_size = len(mag) // N_FFT_BINS
        feat = np.zeros(N_FFT_BINS)
        for i in range(N_FFT_BINS):
            start = i * chunk_size
            end = start + chunk_size
            if end > len(mag): end = len(mag)
            if start < end:
                feat[i] = np.mean(mag[start:end])
    
    features = np.log1p(feat).flatten()

    # 4. 模型推論
    input_tensor = features.reshape(1, -1)
    prediction = model.predict(input_tensor, verbose=0)[0]
    
    label_index = int(np.argmax(prediction))
    score = float(prediction[label_index])
    
    result = RUNTIME_LABELS[label_index] if label_index < len(RUNTIME_LABELS) else "unknown"
    
    # 額外檢查：如果是 Silence 類別，直接忽略
    if result == "silence":
        return result, score, label_index

    print(f"   🗣️ 預測: {result} (信心: {score:.3f})")
    return result, score, label_index

def process_audio(audio_chunk):
    global current_state, recording_buffer, state_timer
    
    # 計算當前區塊能量
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    timestamp = time.time()

    # --- 狀態機邏輯 (模擬 ESP32) ---
    
    if current_state == STATE_IDLE:
        # 監聽模式：等待聲音超過門檻
        if rms > VAD_THRESHOLD:
            # print(f"⚡ 觸發! (RMS: {rms:.4f})")
            current_state = STATE_RECORDING
            recording_buffer = list(audio_chunk) # 將觸發的那一塊也放進去 (保留開頭)
            
    elif current_state == STATE_RECORDING:
        # 錄音模式：拼命收集資料直到滿 0.512s
        recording_buffer.extend(audio_chunk)
        
        if len(recording_buffer) >= WINDOW_SIZE:
            # 緩衝區滿了，開始辨識
            full_audio = np.array(recording_buffer[:WINDOW_SIZE]) # 確保長度精確
            
            result, score, _ = predict_from_audio(full_audio)
            
            if score > CONFIDENCE_THRESHOLD and result != "silence":
                print(f"\n>>> 🎉 偵測成功: 【 {result} 】 <<<\n")
            
            # 進入冷卻，避免重複觸發
            current_state = STATE_COOLDOWN
            state_timer = timestamp
            recording_buffer = [] # 清空

    elif current_state == STATE_COOLDOWN:
        # 冷卻模式：休息一下
        if (timestamp - state_timer) > COOLDOWN_TIME:
            current_state = STATE_IDLE
            # print("... 準備就緒 ...")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    process_audio(indata.flatten())

def run_file_mode(input_path, recursive=False):
    # File mode 保持不變，因為檔案通常已經是切好的
    files = []
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        pattern = "**/*.wav" if recursive else "*.wav"
        files = glob.glob(os.path.join(input_path, pattern), recursive=recursive)

    print(f"🔎 測試 {len(files)} 個檔案...")
    for f in files:
        y, sr = librosa.load(f, sr=SAMPLE_RATE)
        # 檔案模式直接預測，不做 VAD 切割，假設檔案就是單字
        predict_from_audio(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "file"], default="live")
    parser.add_argument("--input", help="輸入檔案路徑")
    parser.add_argument("--device", help="裝置 ID")
    args = parser.parse_args()

    global model, input_shape, MODEL_PATH
    
    print(f"⏳ 載入模型: {MODEL_PATH} ...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        input_shape = model.input_shape[1]
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    if args.mode == "file":
        if not args.input:
            print("❌ File mode 需要 --input")
            return
        run_file_mode(args.input)
    else:
        # Live Mode
        try:
            device_id = int(args.device) if args.device else None
            with sd.InputStream(callback=audio_callback, channels=1, 
                              samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, 
                              device=device_id):
                print("🎧 系統啟動! 請對著麥克風說話...")
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 停止")
        except Exception as e:
            print(f"\n❌ 麥克風啟動失敗: {e}")

if __name__ == "__main__":
    main()

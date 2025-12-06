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
# 請確保這些參數跟 train_model.py 訓練時完全一致
# MODEL_PATH = "kws_model.h5"
MODEL_PATH = "kws_model_mixed_finetuned_2.h5"
# 預設標籤順序（建議與訓練時相同）
# 對應 COMMANDS: on -> "開啟", off -> "關閉", silence -> ""
LABELS = ["on", "off", "silence"]
# Runtime label order (can be overridden via CLI --labels)
RUNTIME_LABELS = LABELS.copy()

VAD_THRESHOLD = 0.007    # 音量門檻 (RMS)，需根據麥克風靈敏度調整 (0.005 ~ 0.05)

# Onset/offset detection for complete phoneme capture
# State: are we currently in a "speech burst"?
IN_SPEECH = False
SPEECH_START_TIME = None
SILENCE_START_TIME = None
# RMS thresholds: onset triggers at ONSET_THRESHOLD, offset confirmed after SILENCE_DURATION below OFFSET_THRESHOLD
ONSET_THRESHOLD = VAD_THRESHOLD
OFFSET_THRESHOLD = VAD_THRESHOLD * 0.7
SILENCE_DURATION = 0.08  # 80ms of silence below offset threshold = end of phoneme
# Minimum phoneme duration to accept
MIN_PHONEME_DURATION = 0.04  # 40ms minimum
SAMPLE_RATE = 16000
DURATION = 0.75         # 模型訓練時的長度 (秒)
N_MELS = 16             # 特徵數量 (記得改成你訓練時用的數值，如 16 或 20)
N_FFT_BINS = 64         # 如果使用 FFT-based 特徵，保留低頻前 N 項
# 與訓練時相同的 RMS 正規化設定（若訓練時啟用，推論也應啟用）
NORMALIZE_RMS = True
TARGET_RMS = 0.10
MAX_GAIN = 10.0
RMS_EPS = 1e-8

# --- VAD 與緩衝區設定 ---
BLOCK_DURATION = 0.2    # 每次從麥克風讀取的小片段 (0.1秒)
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
WINDOW_SIZE = int(SAMPLE_RATE * DURATION) # 緩衝區總長度 (要餵給模型的長度)

CONFIDENCE_THRESHOLD = 0.75 # AI 信心門檻
COOLDOWN_TIME = 0.75     # 觸發後休息幾秒，避免同一個字被重複辨識

# --- 載入模型 ---
print("⏳ 載入模型中...")
model = tf.keras.models.load_model(MODEL_PATH)
# 預先讀取 input shape 確保正確
input_shape = model.input_shape[1] 

# --- 全域變數 ---
# 使用 deque (雙端佇列) 實作 Ring Buffer，設定最大長度，舊資料會自動被擠出去
audio_buffer = collections.deque(maxlen=WINDOW_SIZE)
last_trigger_time = 0

print(f"=== 智慧語音偵測 (VAD Mode) ===")
print(f"🎤 麥克風區塊大小: {BLOCK_DURATION}s")
print(f"🧠 模型視窗大小: {DURATION}s")
print(f"⚡ VAD 門檻: {VAD_THRESHOLD}")



def predict_from_audio(full_audio: np.ndarray):
    """Run feature extraction and model prediction on a full audio array (WINDOW_SIZE samples expected).

    Returns (label, score, label_index)
    """
    # Ensure correct length: pad or trim to WINDOW_SIZE
    if len(full_audio) < WINDOW_SIZE:
        pad = np.zeros(WINDOW_SIZE - len(full_audio), dtype=full_audio.dtype)
        full_audio = np.concatenate([pad, full_audio])
    else:
        full_audio = full_audio[-WINDOW_SIZE:]

    # 和訓練一致：先做 RMS 正規化（避免 gTTS/edge-tts 或錄音能量差異）
    if NORMALIZE_RMS:
        cur_rms = np.sqrt(np.mean(full_audio**2)) if full_audio.size > 0 else 0.0
        if cur_rms > RMS_EPS:
            gain = TARGET_RMS / cur_rms
            if gain > MAX_GAIN:
                gain = MAX_GAIN
            if abs(gain - 1.0) > 1e-6:
                full_audio = full_audio * gain
        full_audio = np.clip(full_audio, -1.0, 1.0)

    # FFT-based features (keep low-frequency bins)
    window = np.hanning(len(full_audio))
    spectrum = np.fft.rfft(window * full_audio)
    mag = np.abs(spectrum)
    if len(mag) < N_FFT_BINS:
        mag = np.pad(mag, (0, N_FFT_BINS - len(mag)))
    else:
        mag = mag[:N_FFT_BINS]
    features = np.log1p(mag).flatten()

    # Padding/Trimming 確保形狀吻合
    if len(features) < input_shape:
        features = np.pad(features, (0, input_shape - len(features)))
    else:
        features = features[:input_shape]

    input_tensor = features.reshape(1, -1)
    prediction = model.predict(input_tensor, verbose=0)
    score = float(np.max(prediction))
    label_index = int(np.argmax(prediction))
    # use runtime label ordering (may be overridden via CLI)
    result = RUNTIME_LABELS[label_index] if label_index < len(RUNTIME_LABELS) else LABELS[label_index]
    print(f"   預測結果: {result} (信心指數: {score:.3f})")
    return result, score, label_index


def run_file_mode(input_path: str, recursive: bool = False):
    """Run inference over files in `input_path`. If `input_path` is a file, run it; if a folder, run all wav files inside."""
    files = []
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        pattern = "**/*.wav" if recursive else "*.wav"
        files = glob.glob(os.path.join(input_path, pattern), recursive=recursive)

    if not files:
        print(f"沒有找到音訊檔案: {input_path}")
        return

    print(f"🔎 準備對 {len(files)} 個檔案進行推論...")
    for f in files:
        try:
            y, sr = librosa.load(f, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"   ⚠️ 無法讀取檔案 {f}: {e}")
            continue

        # If audio longer than WINDOW_SIZE, take the last WINDOW_SIZE samples (similar to live buffer)
        if len(y) >= WINDOW_SIZE:
            segment = y[-WINDOW_SIZE:]
        else:
            # pad
            pad = np.zeros(WINDOW_SIZE - len(y), dtype=y.dtype)
            segment = np.concatenate([pad, y])

        # predict_from_audio 內會進行相同的 RMS 正規化與特徵抽取
        result, score, idx = predict_from_audio(segment)
        print(f"{os.path.basename(f)} -> {result} (score: {score:.3f})")


def process_audio(audio_chunk):
    global last_trigger_time, IN_SPEECH, SPEECH_START_TIME, SILENCE_START_TIME
    
    # 1. 將新資料放入緩衝區 (Ring Buffer)
    audio_buffer.extend(audio_chunk)
    
    # 如果緩衝區還沒滿 (剛啟動時)，就先跳過
    if len(audio_buffer) < WINDOW_SIZE:
        return

    # 2. 計算能量 (RMS - Root Mean Square)
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    current_time = time.time()
    
    # 3. Onset/Offset detection logic
    # Detect when a phoneme STARTS (RMS crosses ONSET_THRESHOLD upward)
    if not IN_SPEECH and rms > ONSET_THRESHOLD:
        IN_SPEECH = True
        SPEECH_START_TIME = current_time
        SILENCE_START_TIME = None
    
    # If currently in speech and RMS falls below offset threshold, start silence timer
    if IN_SPEECH:
        if rms <= OFFSET_THRESHOLD:
            if SILENCE_START_TIME is None:
                SILENCE_START_TIME = current_time
            # If silence persisted long enough, phoneme is complete
            if (current_time - SILENCE_START_TIME) >= SILENCE_DURATION:
                # Phoneme ended: get full audio from ring buffer and predict
                duration = current_time - SPEECH_START_TIME
                if duration >= MIN_PHONEME_DURATION:
                    full_audio = np.array(audio_buffer)
                    result, score, label_index = predict_from_audio(full_audio)
                    
                    # Direct trigger: if confident and not silence, print result
                    if score > CONFIDENCE_THRESHOLD and result not in ("silence", "other"):
                        if (current_time - last_trigger_time) > COOLDOWN_TIME:
                            print(f"\n>>> 🎯 偵測到: 【 {result} 】 (信心度: {score:.2f})")
                            last_trigger_time = current_time
                
                # Reset speech state
                IN_SPEECH = False
                SPEECH_START_TIME = None
                SILENCE_START_TIME = None
        else:
            # RMS rose again during speech, reset silence timer
            SILENCE_START_TIME = None

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    # 將輸入轉為單聲道並交給處理函式
    process_audio(indata.flatten())


def main():
    parser = argparse.ArgumentParser(description="Live demo or file mode for keyword spotting")
    parser.add_argument("--mode", choices=["live", "file"], default="live", help="運行模式: live (麥克風) 或 file (讀檔案進行推論)")
    parser.add_argument("--input", help="輸入檔案或資料夾 (file 模式使用)")
    parser.add_argument("--labels", default=','.join(LABELS), help="以逗號分隔的標籤順序，例: on,off,silence")
    parser.add_argument("--device", help="音訊輸入裝置 id 或名稱（使用 sd.query_devices() 取得）")
    parser.add_argument("--conf-threshold", type=float, default=globals().get('CONFIDENCE_THRESHOLD', 0.75), help="觸發信心門檻 (default: %(default)s)")
    parser.add_argument("path", nargs="?", help="備選的輸入檔案或資料夾（可用於不小心把 `-- input` 寫成兩個 token 的情況）")
    parser.add_argument("--recursive", action="store_true", help="在資料夾中遞迴尋找 WAV 檔案")
    args = parser.parse_args()

    # apply CLI overrides
    global RUNTIME_LABELS, CONFIDENCE_THRESHOLD
    RUNTIME_LABELS = [l.strip() for l in args.labels.split(',') if l.strip()]
    CONFIDENCE_THRESHOLD = float(args.conf_threshold)

    if args.mode == "file":
        # Accept either --input or a positional path (for convenience)
        input_path = args.input or args.path
        if not input_path:
            print("請使用 --input 或直接提供路徑參數，例: --mode file --input <path> 或 --mode file <path>")
            return
        run_file_mode(input_path, recursive=args.recursive)
        return

    # live mode: 印出可用裝置並嘗試打開 InputStream，強化錯誤處理
    print("正在監聽... (按 Ctrl+C 停止)")
    try:
        devices = sd.query_devices()
        default_dev = sd.default.device
        print("音訊裝置數量:", len(devices))
        print("預設裝置 index:", default_dev)
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"  [{i}] {dev['name']} (in-channels={dev['max_input_channels']})")
    except Exception as e:
        print(f"(warn) 無法查詢音訊裝置: {e}")

    # resolve device argument
    chosen_device = None
    if args.device:
        try:
            chosen_device = int(args.device)
        except Exception:
            chosen_device = args.device

    # sanity check: if chosen_device is present, ensure it has input channels
    try:
        if chosen_device is not None:
            dev_info = sd.query_devices(chosen_device)
        else:
            # pick default input device index if available
            default_dev = sd.default.device
            # sd.default.device may be a tuple (in, out) or int
            if isinstance(default_dev, (list, tuple)):
                default_in = default_dev[0]
            else:
                default_in = default_dev
            dev_info = sd.query_devices(default_in)
        if dev_info.get('max_input_channels', 0) == 0:
            print("\n❗ 選定的裝置沒有輸入通道 (max_input_channels=0)。請指定其他輸入裝置。")
            print("可用裝置請先查詢： python -c \"import sounddevice as sd; print(sd.query_devices())\"")
            return
    except Exception as e:
        print(f"(warn) 無法檢查選定裝置資訊: {e}")

    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, device=chosen_device):
            print("🎧 Audio stream started. Callback will be called on incoming frames.")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n🛑 程式停止")
            finally:
                print("Audio stream closed normally.")
    except Exception as e:
        print("\n❌ 無法啟動麥克風輸入流:", e)
        print("請檢查你的音訊裝置是否正確連接或使用 --device 指定裝置 id/名稱。若要查看裝置清單，執行以下 Python 命令：")
        print("  python -c \"import sounddevice as sd; print(sd.query_devices())\"")
        return


if __name__ == "__main__":
    main()

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time

# --- 設定參數 ---
SAMPLE_RATE = 16000  # 16kHz 對語音夠用，且硬體處理負擔比 44.1kHz 小很多
DURATION = 1.0       # 每個指令錄 1 秒
OUTPUT_FOLDER = "dataset"
LABELS = ["kai", "qi", "guan", "bi", "silence"]  # 指令列表
NUM_SAMPLES = 15     # 每個指令錄 20 次 (MVP 快速驗證用)
NORMALIZE = True     # 是否自動放大/正規化錄音
TARGET_PEAK = 0.9    # 目標峰值 (0..1)，接近 1.0 會較大聲
MAX_GAIN = 20.0      # 最大放大倍數，避免把噪音放很大

def record_audio(filename):
    print(f"🎙️ 錄音中... ({filename})")
    try:
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    except Exception as e:
        print(f"   ⚠️ 無法開始錄音: {e}")
        return False
    sd.wait()  # 等待錄音結束
    # 計算峰值與 RMS, 以利診斷
    flat = recording.flatten()
    peak = float(np.max(np.abs(flat))) if flat.size > 0 else 0.0
    rms = float(np.sqrt(np.mean(flat**2))) if flat.size > 0 else 0.0
    def to_db(x):
        return 20.0 * np.log10(x) if x > 0 else -999.0

    print(f"   峰值: {peak:.6f} ({to_db(peak):.1f} dBFS), RMS: {rms:.6f} ({to_db(rms):.1f} dBFS)")

    # 如果峰值過低且啟用正規化，嘗試放大到目標峰值 (受 MAX_GAIN 限制)
    if NORMALIZE and peak > 0.0:
        gain = TARGET_PEAK / peak
        if gain > MAX_GAIN:
            print(f"   ⚠️ 計算出過高的增益 ({gain:.1f}x)，限制為 {MAX_GAIN}x")
            gain = MAX_GAIN
        if gain > 1.0:
            print(f"   🔊 應用增益: {gain:.2f}x")
            recording = (recording * gain).astype('float32')
            # 重新計算峰值、RMS
            flat = recording.flatten()
            peak = float(np.max(np.abs(flat))) if flat.size > 0 else 0.0
            rms = float(np.sqrt(np.mean(flat**2))) if flat.size > 0 else 0.0
            print(f"   新峰值: {peak:.6f} ({to_db(peak):.1f} dBFS), 新RMS: {rms:.6f} ({to_db(rms):.1f} dBFS)")

    # 存檔 (轉為 16-bit PCM，這是硬體最常用的格式)
    try:
        wav.write(filename, SAMPLE_RATE, (recording * 32767).astype(np.int16))
        print("✅ 完成")
        return True
    except Exception as e:
        print(f"   ⚠️ 存檔失敗: {e}")
        return False

# --- 主流程 ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("=== 開始錄製資料集 ===")
print("請在看到提示後說出指令。")

# 顯示預設錄音裝置資訊，協助診斷
try:
    default_input = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    dev = sd.query_devices(default_input, kind='input')
    print(f"預設輸入裝置: {dev['name']} (最大通道: {dev['max_input_channels']})")
except Exception:
    # 忽略查詢錯誤
    pass

for label in LABELS:
    label_dir = os.path.join(OUTPUT_FOLDER, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    print(f"\n準備錄製類別: 【 {label.upper()} 】")
    input("按 Enter 鍵開始...")
    
    for i in range(NUM_SAMPLES):

        print(f"[{i+1}/{NUM_SAMPLES}] 1秒後開始...", end="\r")
        time.sleep(1)
        
        filename = os.path.join(label_dir, f"{label}_{i}.wav")
        ok = record_audio(filename)
        if not ok:
            print("   ⚠️ 錄音失敗，請檢查麥克風或裝置設定")
        time.sleep(0.5) # 休息一下

print("\n🎉 資料集錄製完成！")

import os
import glob
import argparse
import numpy as np
import librosa
import soundfile as sf

# --- 針對 ESP32 優化的參數 ---
SAMPLE_RATE = 16000
DURATION = 0.512  # 配合 ESP32 的 8192 buffer

# 變速範圍 (針對非 Silence)
SPEEDS = [0.95, 1.0, 1.05] 

# ⚠️ 改良: 模擬 VAD 觸發延遲 (Pre-padding)
# 我們不再做左右位移，而是模擬「聲音前面有多少空白」
# VAD 通常很靈敏，所以聲音前面只有 0ms ~ 100ms 的空白
VAD_PRE_PAD_LIMIT_MS = 100 

# 噪音係數
NOISE_FACTORS = [0.0, 0.005]

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def load_audio(path, is_silence=False, sr=SAMPLE_RATE):
    try:
        y, _ = librosa.load(path, sr=sr)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros(int(SAMPLE_RATE * DURATION))

    # ⚠️ 關鍵修正：如果是 Silence，絕對不要 Trim！
    # 否則白噪音會被當成靜音切掉，變成空陣列
    if not is_silence:
        y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    
    return y

def process_audio(y, speed, noise_factor, is_silence=False):
    target_len = int(SAMPLE_RATE * DURATION)
    
    # 1. Time Stretch (變速) - 只對人聲做
    if not is_silence and speed != 1.0:
        try:
            y = librosa.effects.time_stretch(y, rate=speed)
        except:
            pass 
            
    # 2. Prepare Canvas (準備畫布)
    canvas = np.zeros(target_len)
    src_len = len(y)
    
    # 3. Alignment (對齊邏輯)
    if is_silence:
        # 如果是 Silence，直接填滿或隨機切
        if src_len >= target_len:
            start = np.random.randint(0, src_len - target_len + 1)
            canvas = y[start : start + target_len]
        else:
            # 不夠長就重複填入 (Loop)
            repeats = (target_len // src_len) + 1
            y_tiled = np.tile(y, repeats)
            canvas = y_tiled[:target_len]
    else:
        # ⚠️ 關鍵修正：模擬 ESP32 VAD 行為 (靠左對齊)
        # VAD 觸發後，聲音通常在 buffer 的很前面
        # 我們隨機在前面加 0 ~ 100ms 的空白
        
        max_pad_samples = int(VAD_PRE_PAD_LIMIT_MS * SAMPLE_RATE / 1000)
        # 確保不會把聲音擠出畫布
        available_space = target_len - src_len
        
        if available_space > 0:
            # 在「0」到「最大允許延遲」之間選一個
            # 同時也不能超過畫布剩餘空間
            limit = min(max_pad_samples, available_space)
            start_pos = np.random.randint(0, limit + 1)
        else:
            start_pos = 0 # 聲音太長，直接從頭放
            
        dst_end = min(target_len, start_pos + src_len)
        src_end = min(src_len, dst_end - start_pos)
        
        canvas[start_pos : start_pos + src_end] = y[0 : src_end]
        
    # 4. Add Noise (加噪音)
    # 如果原本就是 Silence (噪音)，再疊加一點隨機性
    if noise_factor > 0:
        noise = np.random.randn(len(canvas))
        canvas = canvas + noise * noise_factor
        
    # Clip
    canvas = np.clip(canvas, -1.0, 1.0)
    return canvas

def augment_folder(src_dir, out_dir, label_name):
    ensure_dir(out_dir)
    wavs = sorted(glob.glob(os.path.join(src_dir, "*.wav")))
    if not wavs:
        return 0

    # 判斷是否為 silence 資料夾
    is_silence = (label_name == "silence")

    cnt = 0
    for w in wavs:
        base = os.path.splitext(os.path.basename(w))[0]
        y_orig = load_audio(w, is_silence=is_silence)
        
        if len(y_orig) == 0: continue

        # 1. 先產生一個「標準版」(無變速、無額外噪音)
        # 這樣確保原始特徵一定會被學到
        try:
            y_base = process_audio(y_orig, 1.0, 0.0, is_silence=is_silence)
            out_name_orig = f"{base}_orig.wav"
            sf.write(os.path.join(out_dir, out_name_orig), y_base, SAMPLE_RATE)
            cnt += 1
        except Exception as e:
            print(f"Error processing base {w}: {e}")

        # 2. 產生變體
        # 如果是 Silence，我們不需要產生那麼多變體 (因為它本身就是隨機噪音)
        # 只要產生幾個不同音量的版本就好，不然資料會失衡
        
        if is_silence:
            # Silence 簡單擴增：只改變音量 (透過 noise_factor 模擬)
            for _ in range(5): # 每個原始噪音檔產生 5 個變體
                nf = np.random.uniform(0.001, 0.01)
                y_aug = process_audio(y_orig, 1.0, nf, is_silence=True)
                out_name = f"{base}_aug_{cnt}.wav"
                sf.write(os.path.join(out_dir, out_name), y_aug, SAMPLE_RATE)
                cnt += 1
        else:
            # 人聲完整擴增
            for sp in SPEEDS:
                # 產生 2 種不同的 VAD 延遲位置 (透過迴圈跑兩次 process_audio 隨機決定)
                for _ in range(2): 
                    for nf in NOISE_FACTORS:
                        y_aug = process_audio(y_orig, sp, nf, is_silence=False)
                        
                        nf_label = int(nf * 1000)
                        sp_label = int(sp * 100)
                        # 用亂數當後綴避免檔名衝突
                        rnd_id = np.random.randint(0, 9999)
                        
                        out_name = f"{base}_sp{sp_label}_n{nf_label}_{rnd_id}.wav"
                        sf.write(os.path.join(out_dir, out_name), y_aug, SAMPLE_RATE)
                        cnt += 1
    return cnt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="dataset", help="來源資料夾 (generate_audio 的輸出)")
    parser.add_argument("--out", default="dataset_final_train", help="輸出資料夾 (給 finetune 用)")
    parser.add_argument("--trim-db", default=20, type=float, help="Trim dB")
    args = parser.parse_args()

    src = args.src
    out = args.out

    if not os.path.isdir(src):
        print(f"Source folder not found: {src}")
        return

    ensure_dir(out)
    labels = [d for d in sorted(os.listdir(src)) if os.path.isdir(os.path.join(src, d))]

    global TRIM_TOP_DB
    TRIM_TOP_DB = float(args.trim_db)
    
    total = 0
    for lbl in labels:
        print(f"Processing {lbl}...")
        # 傳入 lbl 名稱以便判斷是否為 silence
        n = augment_folder(os.path.join(src, lbl), os.path.join(out, lbl), lbl)
        total += n

    print(f"✅ Done! Generated {total} files in '{out}'.")

if __name__ == '__main__':
    main()

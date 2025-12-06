import os
import glob
import argparse
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

# --- 參數設定 ---
SAMPLE_RATE = 16000
DURATION = 0.512
N_FFT_BINS = 64
NORMALIZE_RMS = True
TARGET_RMS = 0.10
MAX_GAIN = 10.0

def load_and_prepare(path, label, augment=False):
    try:
        y_full, _ = librosa.load(path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return np.zeros(int(SAMPLE_RATE * DURATION))

    # 1. Trim
    # ⚠️ 修改：如果是 silence 類別，不要 Trim！因為它是噪音，Trim 會把它切光光
    if label != "silence":
        y_trimmed, _ = librosa.effects.trim(y_full, top_db=20)
    else:
        y_trimmed = y_full

    target_len = int(SAMPLE_RATE * DURATION)
    
    # ... (Padding 邏輯保持不變) ...
    if len(y_trimmed) > target_len:
        max_offset = len(y_trimmed) - target_len
        if augment and max_offset > 0 and np.random.rand() > 0.2:
            start = np.random.randint(0, max_offset)
        else:
            start = 0
        y = y_trimmed[start : start + target_len]
    else:
        pad_len = target_len - len(y_trimmed)
        max_front_pad = pad_len // 4
        if augment and max_front_pad > 0:
             front_pad = np.random.randint(0, max_front_pad)
        else:
             front_pad = 0
        back_pad = pad_len - front_pad
        y = np.pad(y_trimmed, (front_pad, back_pad), mode='constant')

    # 2. Noise Injection
    if augment:
        noise_level = np.random.uniform(0.005, 0.02)
        noise = np.random.randn(len(y)) * noise_level
        y = y + noise

    # 3. RMS Normalization
    # ⚠️ 關鍵修改：如果是 silence，不要放大音量！
    # 我們希望 silence 保持原本小聲的狀態，或者隨機給一個很小的音量
    if NORMALIZE_RMS:
        eps = 1e-8
        if y.size > 0:
            cur_rms = np.sqrt(np.mean(y**2))
        else:
            cur_rms = 0.0
            
        if cur_rms > eps:
            # 如果是 silence，我們不放大到 0.1，而是保持原樣 (或者限制上限)
            if label == "silence":
                # 讓噪音保持在 0.02 以下，不要跟人聲(0.1)一樣大
                target = np.random.uniform(0.005, 0.02) 
                gain = target / cur_rms
            else:
                gain = TARGET_RMS / cur_rms
                
            if gain > MAX_GAIN: gain = MAX_GAIN
            y = y * gain
        y = np.clip(y, -1.0, 1.0)
        
    return y

def extract_features_from_wave(y):
    # 確保長度正確 (雙重保險)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]

    window = np.hanning(len(y))
    spectrum = np.fft.rfft(window * y)
    mag = np.abs(spectrum)
    
    chunk_size = len(mag) // N_FFT_BINS
    feat = np.zeros(N_FFT_BINS)
    
    for i in range(N_FFT_BINS):
        start = i * chunk_size
        end = start + chunk_size
        if end > len(mag): end = len(mag)
        
        if start < end:
            feat[i] = np.mean(mag[start:end])
        else:
            feat[i] = 0.0

    feat = np.log1p(feat)
    return feat.flatten()

def build_dataset(data_dir, labels):
    X = []
    y = []
    groups = []
    
    print("📂 Loading dataset...")
    for i, label in enumerate(labels):
        label_dir = os.path.join(data_dir, label)
        # ... (檢查資料夾) ...
        files = glob.glob(os.path.join(label_dir, "*.wav"))
        
        for f in files:
            base_id = os.path.splitext(os.path.basename(f))[0]
            try:
                # ⚠️ 傳入 label 參數
                wave_orig = load_and_prepare(f, label, augment=False)
                X.append(extract_features_from_wave(wave_orig))
                y.append(i)
                groups.append(base_id)
                
                # Augment
                for _ in range(3):
                    wave_aug = load_and_prepare(f, label, augment=True)
                    X.append(extract_features_from_wave(wave_aug))
                    y.append(i)
                    groups.append(base_id)
            except Exception as e:
                print(f"Error: {e}")
                continue
    return np.array(X), np.array(y), np.array(groups)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset_final_train", help="資料資料夾")
    parser.add_argument("--out", default="kws_model_finetuned.h5", help="輸出模型路徑")
    args = parser.parse_args()

    # 標籤定義
    labels = ["kai", "qi", "guan", "bi", "silence"]
    
    X, y_indices, groups = build_dataset(args.data, labels)
    
    if len(X) == 0:
        print("❌ Error: No data found. Please check your dataset folder.")
        return

    print(f"📊 Dataset built. Shape: {X.shape}")

    # Split by group (original file base) to avoid leakage between augmented variants
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_indices, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_indices[train_idx], y_indices[test_idx]

    # 計算權重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print("⚖️ Class Weights:", class_weight_dict)

    # 模型結構
    feature_dim = X.shape[1]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(feature_dim,)),
        tf.keras.layers.Dense(64, activation='relu'), 
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(labels), activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("🚀 Training...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=40, batch_size=32, 
              class_weight=class_weight_dict)

    model.save(args.out)
    print(f"✅ Model saved to {args.out}")

if __name__ == '__main__':
    main()

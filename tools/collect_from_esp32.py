import serial
import serial.tools.list_ports
import numpy as np
import time

PORT = "COM3"      # 改成你的 ESP32 埠
BAUD = 115200
N_BINS = 64

LABELS = {
    "k": ("kai", 0),
    "q": ("qi", 1),
    "g": ("guan", 2),
    "b": ("bi", 3),
    "s": ("silence", 4),
}

def parse_feat_line(line: str):
    """
    line 格式: FEAT:-1,f0,f1,...,f63
    回傳: features_array 或 None
    """
    if not line.startswith("FEAT:"):
        return None
    line = line[5:].strip()
    parts = line.split(",")
    if len(parts) < N_BINS + 1:
        return None
    # 前面的 -1 先略過
    feats = np.array([float(x) for x in parts[1:1+N_BINS]], dtype=np.float32)
    return feats

def main():
    print("可用 Serial Ports:")
    for p in serial.tools.list_ports.comports():
        print(" ", p.device, "-", p.description)

    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    print(f"已連線 {PORT}\n")

    # 1️⃣ 問一次要錄哪個 label
    print("要錄哪一個指令？輸入代號：")
    print("  k = kai")
    print("  q = qi")
    print("  g = guan")
    print("  b = bi")
    print("  s = silence")
    key = input("請輸入(k/q/g/b/s)：").strip().lower()

    if key not in LABELS:
        print("❌ 無效的標籤代號，結束程式")
        return

    label_name, label_id = LABELS[key]
    print(f"\n👉 現在開始收【{label_name}】樣本")
    print("   對著 ESP32 一直念這個字，錄完按 Ctrl+C 結束\n")

    all_features = []
    all_labels = []

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            feats = parse_feat_line(line)
            if feats is None:
                continue

            all_features.append(feats)
            all_labels.append(label_id)

            n = len(all_features)
            if n % 10 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] 已收 {n} 筆 {label_name} 樣本")

    except KeyboardInterrupt:
        print("\n⏹ 停止錄製")

    ser.close()

    if len(all_features) == 0:
        print("⚠️ 沒有收集到任何樣本，直接結束")
        return

    X = np.stack(all_features, axis=0)
    y = np.array(all_labels, dtype=np.int64)

    print("總共收集樣本:", X.shape, "標籤:", y.shape)

    out_name = f"esp32_kws_dataset_{label_name}.npz"
    np.savez(out_name, X=X, y=y)
    print(f"✅ 已存成 {out_name}")

if __name__ == "__main__":
    main()

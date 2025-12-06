import serial
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000

def parse_line(prefix, line):
    if not line.startswith(prefix):
        return None
    data_str = line[len(prefix):].strip()
    vals = [float(x) for x in data_str.split(",") if x]
    return np.array(vals, dtype=np.float32)

def main():
    ser = serial.Serial("COM3", 115200, timeout=1)  # 改成你的埠

    raw = None
    mag = None
    feats = None

    print("等待 ESP32 傳 RAW/MAG/FEATURES ...")
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        if line.startswith("RAW:"):
            raw = parse_line("RAW:", line)
            print("Got RAW, len =", len(raw))
        elif line.startswith("MAG:"):
            mag = parse_line("MAG:", line)
            print("Got MAG, len =", len(mag))
        elif line.startswith("FEATURES:"):
            feats = parse_line("FEATURES:", line)
            print("Got FEATURES, len =", len(feats))

        # 一旦三種都有就畫圖一次
        if raw is not None and mag is not None and feats is not None:
            plot_all(raw, mag, feats)
            raw = mag = feats = None  # 如果要再抓下一次，清空

def plot_all(raw, mag, feats):
    t = np.arange(len(raw)) / SAMPLE_RATE
    freqs = np.fft.rfftfreq(len(raw), d=1.0/SAMPLE_RATE)

    plt.figure(figsize=(12, 10))

    # 1. 時域波形
    plt.subplot(3,1,1)
    plt.plot(t, raw)
    plt.title("Raw waveform from ESP32")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 2. FFT magnitude
    plt.subplot(3,1,2)
    # 注意：我們直接用 ESP32 算好的 mag；如果你只拿前 N/2+1 就對應 freqs
    n = len(mag)
    freqs = np.linspace(0, SAMPLE_RATE/2, n)
    plt.semilogy(freqs, mag + 1e-8)
    plt.title("Magnitude Spectrum (from ESP32)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X(f)|")

    # 3. 64 維 features
    plt.subplot(3,1,3)
    plt.bar(np.arange(len(feats)), feats)
    plt.title("64-bin log features (from ESP32)")
    plt.xlabel("Bin index")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

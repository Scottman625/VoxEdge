# record_esp32.ino — 函數與資料處理細節說明

本文檔針對專案中的 `record_esp32.ino` 檔案做逐一說明，包含全域參數、主要函數、資料處理步驟、注意事項與優化建議。

**檔案位置**: `record_esp32.ino`

**高階目的**
- 在 ESP32 上透過 I2S 讀取麥克風資料、以固定滑動視窗計算頻譜特徵（64 維）、並以 CSV-like 行輸出（`FEAT:-1,f0,..,f63`）供 PC 端收集並標注。

**一、主要常數與參數**
- `SAMPLE_RATE = 16000`：採樣率 16 kHz，與常見語音模型相容。
- `WINDOW_SIZE = 8192`：視窗長度 8192 samples（0.512s @16kHz）。這是做 FFT 的輸入長度。
- `STRIDE_SIZE = 2560`：步進（hop）大小 2560 samples（0.16s），決定特徵輸出頻率與重疊程度。
- `N_FFT_BINS = 64`：最後輸出的特徵維度（頻帶數）。
- I2S 腳位（`I2S_SCK`, `I2S_WS`, `I2S_SD`）與 `I2S_PORT`：依硬體板子設定。
- AGC 與門檻：`TARGET_RMS`、`MAX_GAIN`、`MIN_RMS_THRESHOLD` 控制自動增益與背景噪音過濾。

說明要點：視窗與步進的組合決定時間/頻率分辨率與輸出更新率；若需更即時回饋，可減小 `WINDOW_SIZE` 或增大 `STRIDE_SIZE` 的頻率（但會影響頻率解析度）。

**二、重要全域變數**
- `ring_buffer`：環形 buffer（int16_t）用以儲存最近 `WINDOW_SIZE` 的 raw samples。
- `hanning_window`：事先計算好的 Hanning 視窗（float），用於減少 FFT 機率窗溢位。
- `vReal`, `vImag`：供 `ArduinoFFT` 使用的實部與虛部陣列。
- `ring_head`：環形 buffer 的寫入位置索引。
- `samples_newly_added`：自上次輸出以來新增 sample 數量，用以判斷是否到達 `STRIDE_SIZE`。
- `last_rms`：最近一次視窗的 RMS（用於決定是否輸出特徵）。

注意：所有這些陣列使用 heap malloc；在資源有限的板子上需確保成功分配，並可視需要改為 static 全域陣列以避免 heap 碎片化。

**三、函數逐一說明**

- `void i2s_install()`
  - 作用：初始化 ESP32 的 I2S 驅動與腳位設定。
  - 關鍵設定：
    - `sample_rate = SAMPLE_RATE`、`bits_per_sample = 16`、`channel_format = I2S_CHANNEL_FMT_ONLY_LEFT`（只使用左聲道）
    - `dma_buf_count = 8`、`dma_buf_len = 64`：I2S DMA 緩衝設定，影響延遲與穩定度。
  - 注意：`use_apll = false`（若需要更精確時鐘，可改為 true，但需硬體支援）。

- `void precompute_hanning()`
  - 作用：事先計算 Hanning 視窗係數，減少 run-time 運算。
  - 實作：hanning_window[i] = 0.5f * (1 - cos(2π i / (WINDOW_SIZE-1)))。
  - 注意：視窗大小與 FFT 輸出尺度會互相影響，在前處理端與訓練資料使用相同視窗很重要。

- `void extract_features(float* output_features)`
  - 作用：將當前 `WINDOW_SIZE` 的原始 samples 處理為長度為 `N_FFT_BINS` 的頻譜特徵。
  - 完整步驟：
    1. 讀取 `WINDOW_SIZE` 個 sample（用 `ring_head` 做環形索引），並將 int16 轉為 float 范圍 [-1, 1]。
    2. 計算該視窗的 RMS（root-mean-square），存入 `last_rms`。
    3. AGC（自動增益控制）：若 RMS 小於 0.005，固定 gain=1（避免放大噪音）；否則 gain = TARGET_RMS / rms；再 clamp 至 `MAX_GAIN`。
    4. 套用 gain、clip 到 [-1,1]，並乘以 Hanning 視窗；`vImag` 清零。
    5. 執行 FFT（`FFT.compute(..., WINDOW_SIZE, FFT_FORWARD)`），再 call `complexToMagnitude` 得到幅度譜。
    6. 將頻譜（長度 `spectrum_len = WINDOW_SIZE/2 + 1`）分塊（binning）為 `N_FFT_BINS`（每 bin 平均對應 chunk 的能量），產生 64 維向量。
    7. 做簡單的 EQ：第 0 與第 1 個頻帶分別乘 0.5 與 0.75（削弱極低頻影響）。
    8. 對每個頻帶做 `log1p`（log(x+1)）以壓縮動態範圍，減少數值差異。
  - 輸出：填寫 `output_features[0..N_FFT_BINS-1]`。
  - 設計動機與注意：
    - 用 RMS+AGC 使不同音量下的語音特徵可比較，但同時用 `MIN_RMS_THRESHOLD` 跳過背景低能量段，減少大量無用資料。
    - 將高解析度頻譜 binning 成 64 維減少輸出與儲存成本，但會犧牲頻率解析度；bin 的劃分（chunk_size）是簡單平均，若需更準確可用 mel-scale 或對數頻率分割。
    - 使用 `log1p` 而不是 raw magnitude 可提高模型對低能量頻帶的靈敏度，同時抑制極端值。

- `void output_feat_line(const float* feat)`
  - 作用：把特徵向量以簡單的 CSV 行輸出到 `Serial`。
  - 格式：先印出 `FEAT:-1`（此處 label 暫為 -1，由 PC 端後處理標注），接著每個頻帶用逗號分隔，浮點數格式保留 6 位小數。
  - 範例：`FEAT:-1,0.001234,0.002345,...`。
  - 注意：輸出頻率由 `STRIDE_SIZE` 控制；如果 `Serial` 傳輸過慢可能造成緩衝堆積或遺失資料，必要時降低輸出率或增加緩衝/壓縮。

- `void setup()`
  - 作用：啟動 `Serial`、分配記憶體（ring buffer、視窗、FFT 緩衝）、計算 Hanning、初始化 I2S 驅動。
  - 若 `malloc` 失敗會印錯誤並停在迴圈中（保護措施）。

- `void loop()`
  - 作用：持續從 I2S 讀取原始音訊，把 samples 推入 `ring_buffer`，每當新加入的 samples >= `STRIDE_SIZE` 時呼叫 `extract_features`，若 `last_rms` 大於 `MIN_RMS_THRESHOLD` 則用 `output_feat_line` 輸出。
  - I2S 讀取使用 `i2s_read`，每次讀 `i2s_read_len = 1024` bytes（512 samples）；讀到的 bytes 會依序寫入 `ring_buffer`。

**四、資料處理上的特殊處理與設計理由**
- 環形 buffer + 固定視窗（WINDOW_SIZE）與步進（STRIDE_SIZE）：允許連續滑動窗口的特徵擷取，而非每次都等待剛好 `WINDOW_SIZE` 新 samples，能得到重疊且時間上連續的特徵序列。
- RMS 與 `MIN_RMS_THRESHOLD`：避免輸出背景噪音的大量樣本，方便後處理標注與訓練資料清洗。
- AGC（Target RMS）：在現場錄音條件（音量差異大）下，AGC 可使特徵在音量變化時更一致，但會改變信號相對能量分佈，應與訓練資料一致（或在訓練時模擬 AGC）。
- Hanning 視窗：減少視窗邊界的不連續性導致的頻譜洩漏（spectral leakage）。
- FFT 後的簡單 binning：以平均能量劃分頻帶是低成本做法；若想提高語音辨識效果，改為 mel filterbank 或對數頻率分割通常更好。
- log1p：常用於將頻譜能量壓縮成近似對數尺度，有助於模型學習。

**五、潛在問題與優化建議**
- 記憶體：`WINDOW_SIZE=8192` 與多個 float 陣列會耗較多 RAM（approx. 8192*2 bytes for ring + 8192*4*3 bytes for floats ≈ ~131KB+），在較小的 ESP32 variant 需注意可用 heap。
- 實時性：目前流程在每個 `STRIDE_SIZE` 輸出一次，若 `Serial` 傳輸成瓶頸，可能導致資料積壓。可改為降採樣輸出頻率或用壓縮（例如 quantize 特徵到 int16 後傳送）。
- 精度：頻帶 binning 採平均，對高頻細節弱；若模型在某些關鍵頻帶上表現不佳，考慮改為 mel filterbank 或增加 `N_FFT_BINS`。
- AGC 設計：目前對非常小的 RMS（<0.005）不放大，避免放大噪音；如果環境安靜但說話聲小，可能漏掉有效語音。可加入短期能量門檻邏輯或自適應噪音估計。
- FFT library：`ArduinoFFT` 為 float 實作，若想加速可改為 CMSIS DSP（固定小數點/快速 FFT）或使用硬體加速的 FFT。

**六、建議的擴充或實驗**
- 用 mel-filterbank 替換平均 binning，並確認訓練時相同的前處理。
- 在輸出前進行 per-band normalization（例如減去每個 band 的靜默期平均值），提升穩定性。
- 將輸出改為二進位壓縮格式（例如 int16 或 8-bit quantized），減少 `Serial` 帶寬。
- 在 PC 端寫一個 parser 接受 `FEAT:-1,...` 並自動標注（GUI）能加速資料蒐集流程。

---

如需，我可以：
- 把此說明翻成英文；
- 將說明內的建議改寫為具體 patch（例如改為 mel-bin 或改用 static 陣列）；
- 或新增一個 PC 端的小工具範例（解析 `FEAT:` 行並存成 NPZ/CSV）。

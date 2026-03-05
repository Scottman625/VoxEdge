# live_demo.ino — 函數與資料處理細節說明

本文檔針對專案中的 `live_demo.ino`（用於 ESP32 本地推論）做逐一說明，包含各參數、主要函數、片語狀態機、推論流程、資料處理上的特殊設計與優化建議。

**檔案位置**: `live_demo.ino`

**高階目的**
- 在 ESP32 上透過 I2S 讀取麥克風資料，取得滑動視窗頻譜特徵，將該特徵輸入嵌入式 TensorFlow Lite 模型進行即時推論。此範例實作一個簡單的兩字片語控制：`kai+qi` 開啟、`guan+bi` 關閉。

**一、主要常數與參數說明**
- `SAMPLE_RATE = 16000`：採樣率 16 kHz。
- `WINDOW_SIZE = 8192`：視窗長度（8192 samples ≈ 0.512s）。
- `STRIDE_SIZE = 2560`：步進大小（2560 samples ≈ 0.16s），每到達一次步進就執行一次推論。
- `N_FFT_BINS = 64`：頻帶數（模型輸入維度）。
- I2S 腳位：`I2S_SCK`, `I2S_WS`, `I2S_SD`、`I2S_PORT`（依實際硬體設定）。
- 標籤順序 `LABELS[] = {"kai","qi","guan","bi","silence"}`：必須與訓練時模型輸出的索引順序一致。

- AGC / 門檻：
  - `TARGET_RMS = 0.25f`
  - `MAX_GAIN = 25.0f`
  - `MIN_RMS_THRESHOLD = 0.002f`（比 `record_esp32.ino` 更嚴格或可調以適應系統）

- 片語與單字參數：
  - `WORD_SCORE_THRESHOLD = 0.95f`：單字信心門檻（高門檻以降低誤觸）
  - `WORD_DEBOUNCE_MS = 400`：同一字的防抖時間
  - `PHRASE_TIMEOUT_MS = 1500`：等待片語第二字的最大時間

- TFLite 記憶體：`kTensorArenaSize = 40 * 1024`（40KB arena），`tensorArena` 為運行時 arena 空間。

設計要點：高 `WORD_SCORE_THRESHOLD` 與 debounce 可降低誤觸；若模型信心普遍較低，可降閾值或在模型輸出端做平滑/累積置信度。

**二、重要全域變數**
- `ring_buffer`：環形 int16 樣本緩衝，長度 `WINDOW_SIZE`。
- `hanning_window`, `vReal`, `vImag`：FFT 前處理與暫存陣列。
- `ring_head`, `samples_newly_added`, `last_rms`：環形索引、步進計數、最近視窗 RMS。
- 推論狀態：`last_detected_label`, `last_detected_time`（用於去重）。
- TFLite：`tflModel`, `tflInterpreter`, `tflInputTensor`, `tflOutputTensor` 與 `tensorArena`。
- 片語狀態機：`PhraseStateType`、`phrase_state`、`phrase_start_time`。

注意：陣列均以 `malloc` 在 heap 上配置；在資源緊張裝置上考慮改成 static 陣列或檢查可用 heap。

**三、函數逐一說明與行為細節**

- `void i2s_install()`
  - 功能：初始化 ESP32 I2S 驅動與腳位。
  - 重要設定：`sample_rate = SAMPLE_RATE`, `bits_per_sample = 16`, `channel_format = I2S_CHANNEL_FMT_ONLY_LEFT`。
  - DMA 緩衝：`dma_buf_count = 8`, `dma_buf_len = 64`（調整可影響 latency / throughput / stability）。

- `void precompute_hanning()`
  - 功能：計算 Hanning 視窗係數，供 `extract_features` 使用，減少 runtime 運算與頻譜洩漏。

- `void extract_features(float* output_features)`
  - 功能：將 `WINDOW_SIZE` 原始 samples（從 `ring_buffer` 讀出）處理成長度 `N_FFT_BINS` 的頻譜特徵。
  - 步驟詳述：
    1. 從 `ring_buffer` 以環形索引讀出 `WINDOW_SIZE` samples，並把 int16 轉為 float [-1,1]（vReal）。
    2. 計算視窗 RMS，存入 `last_rms`（避免除以 0，低於 1e-8 則設為 1e-8）。
    3. AGC：若 RMS < 0.005 則不放大（gain=1），否則 gain = TARGET_RMS / rms，再 clamp 為 `MAX_GAIN`。
    4. 套用 gain、clip 到 [-1,1]，乘以 Hanning，並把 `vImag` 設 0。
    5. 執行 FFT（`FFT.compute`），再 `complexToMagnitude` 取得幅度譜。
    6. 把頻譜分割為 `N_FFT_BINS` 個 chunk，對每 chunk 做平均（簡單 binning）。
    7. 做 2 頻帶的 EQ（index 0 *0.5, index 1 *0.75），再對每個 band 做 `log1p`。
  - 輸出填寫 `output_features[0..N_FFT_BINS-1]`。
  - 注意：此前處理須與訓練時前處理一致（包括 Hanning、AGC 邏輯與 log1p）。

- `void reset_phrase_state()`
  - 功能：重置片語狀態機（`phrase_state = PHRASE_NONE`，`phrase_start_time = 0`），並在變化時印出訊息。

- `void update_phrase_state_for_first_word(int label_index, float score, unsigned long now)`
  - 功能：當偵測到片語的第一個字（`kai` 或 `guan`）時呼叫，將 `phrase_state` 設為對應的等待狀態並記錄時間。
  - 行為：若已有其他片語在進行，先重置再設置新狀態，並印出偵測訊息。

- `void handle_second_word_if_phrase(int label_index, float score, unsigned long now)`
  - 功能：當偵測到一個字且目前處於等待第二字的片語狀態時，檢查是否與期待的第二字匹配，且是否在 `PHRASE_TIMEOUT_MS` 內。
  - 若匹配：執行對應動作（將 `current_state` ON/OFF、印出成功訊息）；若不匹配或逾時則重置片語狀態。

- `void run_inference()`
  - 功能：進行一次完整的特徵擷取 + TFLite 推論，並依結果更新片語狀態與系統狀態。
  - 步驟：
    1. 呼叫 `extract_features(features)`。
    2. 若 `last_rms < MIN_RMS_THRESHOLD` 則跳過（避免在背景噪音下推論）。
    3. 把 `features` 複製到 `tflInputTensor->data.f`，呼叫 `tflInterpreter->Invoke()`。
    4. 找到 `tflOutputTensor` 中最大分數 `max_score` 與對應 `max_index`。
    5. 若 `max_score < WORD_SCORE_THRESHOLD`，視為無偵測（但會檢查片語逾時）。
    6. 做單字去重（debounce）以避免重複觸發。
    7. 根據 `max_index` 呼叫 `update_phrase_state_for_first_word` 或 `handle_second_word_if_phrase`，或僅檢查片語逾時。
  - 注意：在 `DEBUG_LOG_PROB` 為 true 時，會印出所有類別的機率，用於調試。

- `void setup()`
  - 功能：初始化 `Serial`、分配記憶體、計算 Hanning 視窗、初始化 I2S、載入 TFLite 模型並配置 interpreter（包含 `AllocateTensors()`），以及初始化狀態變數。
  - 重要錯誤檢查：檢查 `malloc`、模型 schema 版本、`AllocateTensors()` 回傳，任何失敗都會停止在 `while(1)` 保護模式。

- `void loop()`
  - 功能：持續呼叫 `i2s_read` 將讀到的 PCM samples 寫入 `ring_buffer`。當 `samples_newly_added >= STRIDE_SIZE` 時呼叫 `run_inference()`。

**四、資料處理上的特殊處理與設計理由**
- Sliding window + stride：使用環形 buffer 與滑動窗口來達成重疊的特徵序列，能提高時間連續性與即時反應。
- RMS + MIN_RMS_THRESHOLD：在極低能量時跳過推論，減少無意義的推論與誤觸。
- AGC：將不同音量的語音標準化到 `TARGET_RMS`，提高在不同音量下的模型穩定性；但 AGC 會改變信號能量分佈，需與訓練資料一致。
- binning（64 bins）與 log1p：以節省運算與記憶體為目的的低成本頻譜壓縮方式；若需更好效果可改用 mel-filterbank。

**五、片語機制細節**
- 片語狀態機分兩步：偵測第一字（`kai` 或 `guan`）→ 設置等待狀態並記錄時間 → 若在 `PHRASE_TIMEOUT_MS` 內偵測到正確第二字（`qi` 或 `bi`）則執行 ON/OFF。
- 去重（debounce）：避免在接續多個 stride 中重複把同一個字報出。實作為：若 `last_detected_label == max_index` 且距離上次時間 < `WORD_DEBOUNCE_MS`，則忽略。

設計上可調參數：若希望更靈活的片語（允許中間嘈雜干擾），可延長 `PHRASE_TIMEOUT_MS` 或實作更複雜的狀態回滾策略。

**六、潛在問題與優化建議**
- 記憶體：`tensorArena` 40KB + FFT / buffer 的 heap 需求，總耗用需在目標 ESP32 上驗證；在 memory constrained 裝置上考慮減少 `WINDOW_SIZE`、降低 `kTensorArenaSize`（若模型更小）或將陣列改為 static/全域常量以避免 heap 分配失敗。
- 延遲與頻寬：每次 `STRIDE_SIZE` 觸發推論，若 `STRIDE_SIZE` 太小會提高 CPU 負擔；若要降低延遲，可減少 `WINDOW_SIZE` 或改用更快的 FFT 實作（如 CMSIS DSP）。
- 模型相容性：模型輸出維度、標籤順序、以及前處理（Hanning、AGC、log1p、binning）務必與訓練時一致。
- 精度提升建議：
  - 把平均 binning 換成 mel-filterbank。
  - 在模型端改成接受較少頻帶但增加時間序列（例如逐步輸入多個 frame 並用 RNN/CNN temporal fusion）。
  - 在輸出前做 per-band normalization 或用 sliding noise floor estimate 做背景去除。
- 效能提升：
  - 將 FFT 換成更快的 fixed-point 實作或硬體加速。
  - 將 `features` 量化（例如 int16）以降低記憶體與輸入寫入帶寬。

**七、建議的擴充範例**
- 將 `// TODO: 開啟動作` 改成呼叫實際函式（例如 `on_activate()` / `on_deactivate()`），以便在不同硬體上重用邏輯。
- 實作一個 debug 模式，當 `DEBUG_LOG_PROB` 開啟時把機率以 JSON 輸出，便於 PC 端即時觀察。
- 在 PC 端加入一個小工具解析 `Serial` 的 log（或直接使用串流通訊），把成功片語存成事件 log 供後續分析。

---

如果你想要我把此檔案的說明翻成英文、或直接依建議產生一個 patch（例如改為 static 陣列、降低 `WINDOW_SIZE`、或把 binning 換成 mel filter），告訴我要哪一項，我會繼續實作。

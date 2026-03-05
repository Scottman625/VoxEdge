# VoxEdge — 專案流程說明

本文件說明本專案的整體流程與常用操作，方便快速上手訓練、測試、轉換模型並部署到 ESP32。

**目錄結構概覽**

- `dataset/`：原始語音資料目錄，按關鍵字分類（如 `bi/`, `guan/`, `kai/`, `qi/`, `silence/`）。
- `dataset_esp32/`：ESP32 用合併後的資料集（`esp32_kws_dataset_merged.npz`）。
- `train/`：訓練腳本（`train_model.py`, `train_esp32_features.py`, `train_finetune.py`）。
- `tools/`：輔助工具（資料增強、檢查、模型轉換、導出等）。
- 模型檔：`kws_model.h5`, `kws_model_finetuned.h5`, `kws_model_esp32.h5`。
- `model.tflite`：TensorFlow Lite 模型（部署到 ESP32 前的格式）。
- `model_data.h` / `model_data.cc`：將 `model.tflite` 轉為 C 陣列，放到嵌入式專案中使用。
- `record_esp32.ino`：範例 Arduino 程式碼，用於 ESP32 偵測與通訊。
- `test/`：簡單測試與範例腳本（`live_demo.py`, `test_esp32_model.py` 等）。

**主要工作流程（高階）**

1. 準備資料
   - 將語音檔放到 `dataset/<label>/`。使用 `tools/check_wavs.py` 檢查檔案品質。
   - 若要為 ESP32 生成特殊格式或合併資料，執行 `tools/augment_expand_esp32_dataset.py` 或 `tools/augment_and_expand_dataset.py`。

2. 訓練模型
   - 一般訓練（初始模型）：

```bash
python train/train_model.py --data-dir dataset --output kws_model.h5
```

   - 生成或提取特徵以供 ESP32 使用（若有必要）：

```bash
python train/train_esp32_features.py --data dataset_esp32 --out esp32_kws_dataset_merged.npz
```

   - 微調模型（選用）：

```bash
python train/train_finetune.py --pretrained kws_model.h5 --data dataset_final_train --output kws_model_finetuned.h5
```

3. 評估與測試
   - 使用 `tools/evaluate_dataset.py` 或 `test/test.py`、`test/test_esp32_model.py` 進行離線評估。
   - 以 `test/live_demo.py` 進行即時偵測展示。

4. 轉換為 TensorFlow Lite
   - 將 Keras 模型轉為 TFLite：

```bash
python tools/convert_model.py --input kws_model.h5 --output model.tflite
```

   - 若需要量化（大小/效能優化），請使用 `tools/export_tflite.py` 的量化選項。

5. 生成嵌入式 C 資料
   - 使用工具將 `model.tflite` 轉為 C 陣列 `model_data.cc` / `model_data.h`：

```bash
python tools/export_tflite.py --tflite model.tflite --out model_data.h
```

6. 部署到 ESP32
   - 在 Arduino / PlatformIO 專案中加入 `model_data.h`，並上傳 `record_esp32.ino` 或自訂程式。
   - 若使用序列通訊或 Web API，可參考 `record_esp32.ino` 中的通訊範例。

7. 現場測試與監控
   - 使用 `test/live_demo2.py` 或自訂的接收端來接收 ESP32 回傳的辨識結果並顯示。

**常用工具說明**

- `tools/check_wavs.py`：檢查 WAV 檔的格式與一致性。
- `tools/augment_and_expand_dataset.py`：對原始語音資料進行增強（時間拉伸、噪音混入等）。
- `tools/augment_expand_esp32_dataset.py`：為 ESP32 調整/擴充資料集。
- `tools/convert_model.py`：將 Keras 模型轉為 TFLite。
- `tools/export_tflite.py`：將 TFLite 轉為 C 陣列或進行量化處理。
- `tools/plot_from_serial.py`：將 ESP32 串列輸出繪圖（用於偵錯）。

**快速上手（建議流程）**

1. 準備數據：放入 `dataset/` 對應資料夾。
2. 檢查 WAV 檔：

```bash
python tools/check_wavs.py dataset
```

3. 訓練模型（快速測試）：

```bash
python train/train_model.py --data-dir dataset --output kws_model.h5
```

4. 轉換並匯出給 ESP32：

```bash
python tools/convert_model.py --input kws_model.h5 --output model.tflite
python tools/export_tflite.py --tflite model.tflite --out model_data.h
```

5. 把 `model_data.h` 加入 ESP32 專案，並上傳 `record_esp32.ino`。

**進階提示**

- 若要改善在嵌入式裝置上的準確率，優先考慮資料增強與微調（`train_finetune.py`），再做量化。
- 在轉換前在主機上使用 `test/test_esp32_model.py` 驗證輸入前處理（例如 MFCC）與模型輸入的相容性。

---

如果你想要我把 README 翻譯成英文、加入更多範例、或自動產生一個簡單的部署腳本，告訴我想要的項目，我可以繼續擴充。

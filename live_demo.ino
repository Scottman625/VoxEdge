#include <driver/i2s.h>
#include <arduinoFFT.h>
#include <math.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// ================= 1. 參數設定 =================
#define SAMPLE_RATE 16000
#define WINDOW_SIZE 8192    // 0.512s
#define STRIDE_SIZE 2560    // 0.16s
#define N_FFT_BINS 64

// I2S 腳位
#define I2S_SCK 14
#define I2S_WS  15
#define I2S_SD  32
#define I2S_PORT I2S_NUM_0

// 標籤 (要跟訓練順序一致)
const char* LABELS[] = {"kai", "qi", "guan", "bi", "silence"};
#define NUM_LABELS 5

#define IDX_KAI     0
#define IDX_QI      1
#define IDX_GUAN    2
#define IDX_BI      3
#define IDX_SILENCE 4

// ===== AGC / 音量門檻 =====
#define TARGET_RMS 0.25f
#define MAX_GAIN 25.0f
#define MIN_RMS_THRESHOLD 0.002f

// ===== 指令 / 片語 參數 =====
#define WORD_SCORE_THRESHOLD   0.95f   // 單字信心門檻
#define WORD_DEBOUNCE_MS       400     // 同一個字 Debounce
#define PHRASE_TIMEOUT_MS      1500    // 片語第二個字時間限制

// 是否輸出每窗機率 (debug)
#define DEBUG_LOG_PROB false

// 系統 ON/OFF 狀態
enum SystemState {
  STATE_OFF = 0,
  STATE_ON  = 1
};
SystemState current_state = STATE_OFF;

// 片語狀態機
enum PhraseStateType {
  PHRASE_NONE = 0,
  PHRASE_OPEN_WAIT_QI,     // kai -> 等 qi
  PHRASE_CLOSE_WAIT_BI     // guan -> 等 bi
};
PhraseStateType phrase_state = PHRASE_NONE;
unsigned long phrase_start_time = 0;   // 記錄第一個字出現時間

// ================= 2. 全域變數 =================
int16_t *ring_buffer;
float *hanning_window;
float *vReal;
float *vImag;

int ring_head = 0;
int samples_newly_added = 0;
float last_rms = 0.0f;

// 單字去重
int last_detected_label = -1;
unsigned long last_detected_time = 0;

// TFLite
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;
const int kTensorArenaSize = 40 * 1024;
uint8_t tensorArena[kTensorArenaSize];

ArduinoFFT<float> FFT = ArduinoFFT<float>();

// ================= I2S 初始化 =================
void i2s_install() {
  const i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 64,
      .use_apll = false
  };
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);

  const i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_SD
  };
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_zero_dma_buffer(I2S_PORT);
}

// ================= Hanning Window =================
void precompute_hanning() {
  for (int i = 0; i < WINDOW_SIZE; i++) {
    hanning_window[i] = 0.5f * (1.0f - cos(2.0f * PI * i / (WINDOW_SIZE - 1)));
  }
}

// ================= 特徵擷取 =================
void extract_features(float* output_features) {
  float sum_sq = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    int idx = (ring_head + i) % WINDOW_SIZE;
    float val = ring_buffer[idx] / 32768.0f;
    vReal[i] = val;
    sum_sq += val * val;
  }

  float rms = sqrt(sum_sq / WINDOW_SIZE);
  if (rms < 1e-8f) rms = 1e-8f;
  last_rms = rms;

  float gain;
  if (rms < 0.005f) gain = 1.0f;
  else gain = TARGET_RMS / rms;
  if (gain > MAX_GAIN) gain = MAX_GAIN;

  for (int i = 0; i < WINDOW_SIZE; i++) {
    float val = vReal[i] * gain;
    if (val > 1.0f)  val = 1.0f;
    if (val < -1.0f) val = -1.0f;
    vReal[i] = val * hanning_window[i];
    vImag[i] = 0.0f;
  }

  FFT.compute(vReal, vImag, WINDOW_SIZE, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, WINDOW_SIZE);
  int spectrum_len = WINDOW_SIZE / 2 + 1;

  int chunk_size = spectrum_len / N_FFT_BINS;
  for (int i = 0; i < N_FFT_BINS; i++) {
    int start = i * chunk_size;
    int end   = start + chunk_size;
    if (end > spectrum_len) end = spectrum_len;
    if (start >= end) {
      output_features[i] = 0.0f;
      continue;
    }
    float sum = 0.0f;
    for (int j = start; j < end; j++) sum += vReal[j];
    output_features[i] = sum / (end - start);
  }

  output_features[0] *= 0.5f;
  output_features[1] *= 0.75f;

  for (int i = 0; i < N_FFT_BINS; i++) {
    output_features[i] = logf(output_features[i] + 1.0f);
  }
}

// ================= 片語狀態機處理 =================
void reset_phrase_state() {
  if (phrase_state != PHRASE_NONE) {
    Serial.println("ℹ 片語逾時或被打斷，重置。");
  }
  phrase_state = PHRASE_NONE;
  phrase_start_time = 0;
}

void update_phrase_state_for_first_word(int label_index, float score, unsigned long now) {
  // 第一個字：kai 或 guan
  if (phrase_state != PHRASE_NONE) {
    reset_phrase_state();
  }

  if (label_index == IDX_KAI) {
    phrase_state = PHRASE_OPEN_WAIT_QI;
    phrase_start_time = now;
    Serial.printf("🔊 已偵測到『kai』（%.2f），等待『qi』完成【開啟】片語...\n", score);
  } else if (label_index == IDX_GUAN) {
    phrase_state = PHRASE_CLOSE_WAIT_BI;
    phrase_start_time = now;
    Serial.printf("🔊 已偵測到『guan』（%.2f），等待『bi』完成【關閉】片語...\n", score);
  }
}

void handle_second_word_if_phrase(int label_index, float score, unsigned long now) {
  // 若目前沒有在等第二個字，直接跳過
  if (phrase_state == PHRASE_NONE) return;

  // 檢查是否逾時
  if (now - phrase_start_time > PHRASE_TIMEOUT_MS) {
    reset_phrase_state();
    return;
  }

  if (phrase_state == PHRASE_OPEN_WAIT_QI && label_index == IDX_QI) {
    // kai + qi => 開啟
    phrase_state = PHRASE_NONE;
    phrase_start_time = 0;

    if (current_state == STATE_OFF) {
      current_state = STATE_ON;
      Serial.printf("✅ 片語偵測：開啟（kai + qi，%.2f）\n", score);
      // TODO: 開啟動作，例如 digitalWrite(LED_PIN, HIGH);
    } else {
      Serial.println("↻ 已經是開啟狀態，開啟片語略過。");
    }
  } else if (phrase_state == PHRASE_CLOSE_WAIT_BI && label_index == IDX_BI) {
    // guan + bi => 關閉
    phrase_state = PHRASE_NONE;
    phrase_start_time = 0;

    if (current_state == STATE_ON) {
      current_state = STATE_OFF;
      Serial.printf("✅ 片語偵測：關閉（guan + bi，%.2f）\n", score);
      // TODO: 關閉動作，例如 digitalWrite(LED_PIN, LOW);
    } else {
      Serial.println("↻ 已經是關閉狀態，關閉片語略過。");
    }
  } else {
    // 第二個字不是預期的，視為打斷
    reset_phrase_state();
  }
}

// ================= 推論 =================
void run_inference() {
  float features[N_FFT_BINS];
  extract_features(features);

  if (last_rms < MIN_RMS_THRESHOLD) return;

  for (int i = 0; i < N_FFT_BINS; i++) {
    tflInputTensor->data.f[i] = features[i];
  }

  if (tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("❌ Invoke failed!");
    return;
  }

  float* results = tflOutputTensor->data.f;
  float max_score = 0.0f;
  int max_index = 0;
  for (int i = 0; i < NUM_LABELS; i++) {
    if (results[i] > max_score) {
      max_score = results[i];
      max_index = i;
    }
  }

  unsigned long now = millis();

  if (DEBUG_LOG_PROB) {
    Serial.printf("[%lu ms] ", now);
    for (int i = 0; i < NUM_LABELS; i++) {
      Serial.printf("%s: %.2f  ", LABELS[i], results[i]);
    }
    Serial.printf("=> %s (%.2f)  [STATE=%s, PHRASE_STATE=%d]\n",
                  LABELS[max_index], max_score,
                  (current_state == STATE_ON ? "ON" : "OFF"),
                  (int)phrase_state);
  }

  // 信心度不夠，當作沒有字
  if (max_score < WORD_SCORE_THRESHOLD) {
    // 檢查片語是否逾時
    if (phrase_state != PHRASE_NONE &&
        now - phrase_start_time > PHRASE_TIMEOUT_MS) {
      reset_phrase_state();
    }
    return;
  }

  // 單字防抖
  if (last_detected_label == max_index) {
    unsigned long dt = now - last_detected_time;
    if (dt < WORD_DEBOUNCE_MS) {
      // 不印一堆東西，單純略過即可
      return;
    }
  }
  last_detected_label = max_index;
  last_detected_time = now;

  // 第一個字：kai 或 guan
  if (max_index == IDX_KAI || max_index == IDX_GUAN) {
    update_phrase_state_for_first_word(max_index, max_score, now);
  }
  // 第二個字：qi 或 bi
  else if (max_index == IDX_QI || max_index == IDX_BI) {
    handle_second_word_if_phrase(max_index, max_score, now);
  } else {
    // silence 或其他：只做片語逾時檢查
    if (phrase_state != PHRASE_NONE &&
        now - phrase_start_time > PHRASE_TIMEOUT_MS) {
      reset_phrase_state();
    }
  }
}

// ================= Setup =================
void setup() {
  Serial.begin(115200);

  ring_buffer = (int16_t*) malloc(WINDOW_SIZE * sizeof(int16_t));
  hanning_window = (float*) malloc(WINDOW_SIZE * sizeof(float));
  vReal = (float*) malloc(WINDOW_SIZE * sizeof(float));
  vImag = (float*) malloc(WINDOW_SIZE * sizeof(float));

  if (!ring_buffer || !hanning_window || !vReal || !vImag) {
    Serial.println("❌ Malloc Failed!");
    while (1) { delay(1000); }
  }

  precompute_hanning();
  i2s_install();

  tflModel = tflite::GetModel(g_model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Model schema version not matched!");
    while (1) { delay(1000); }
  }

  static tflite::MicroInterpreter static_interpreter(
      tflModel, tflOpsResolver, tensorArena, kTensorArenaSize, &tflErrorReporter);
  tflInterpreter = &static_interpreter;

  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ AllocateTensors() failed");
    while (1) { delay(1000); }
  }

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  current_state = STATE_OFF;
  phrase_state = PHRASE_NONE;
  last_detected_label = -1;
  last_detected_time = 0;

  Serial.println("🚀 ESP32 KWS：兩字片語（kai+qi=開啟, guan+bi=關閉）已啟動");
}

// ================= Loop =================
void loop() {
  const size_t i2s_read_len = 1024;
  char i2s_read_buff[i2s_read_len];
  size_t bytes_read;

  i2s_read(I2S_PORT, (void*)i2s_read_buff, i2s_read_len, &bytes_read, portMAX_DELAY);

  int16_t* samples = (int16_t*)i2s_read_buff;
  int count = bytes_read / 2;

  for (int i = 0; i < count; i++) {
    ring_buffer[ring_head] = samples[i];
    ring_head = (ring_head + 1) % WINDOW_SIZE;
    samples_newly_added++;
  }

  if (samples_newly_added >= STRIDE_SIZE) {
    samples_newly_added = 0;
    run_inference();
  }
}

#include <driver/i2s.h>
#include <arduinoFFT.h>
#include <math.h>

// ================= 1. 參數設定 =================
#define SAMPLE_RATE   16000
#define WINDOW_SIZE   8192    // 0.512s
#define STRIDE_SIZE   2560    // 0.16s
#define N_FFT_BINS    64

// I2S 腳位（依你的板子調整）
#define I2S_SCK   14
#define I2S_WS    15
#define I2S_SD    32
#define I2S_PORT  I2S_NUM_0

// AGC / 門檻
#define TARGET_RMS         0.25f
#define MAX_GAIN           25.0f
#define MIN_RMS_THRESHOLD  0.005f    // 小於這個就當背景不輸出特徵

// ================= 2. 全域變數 =================
int16_t *ring_buffer;
float   *hanning_window;
float   *vReal;
float   *vImag;

int   ring_head = 0;
int   samples_newly_added = 0;
float last_rms = 0.0f;

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
      .ws_io_num  = I2S_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num  = I2S_SD
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
  // A. 轉為 -1~1，並計算 RMS
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

  // B. AGC
  float gain;
  if (rms < 0.005f) {
    gain = 1.0f;
  } else {
    gain = TARGET_RMS / rms;
  }
  if (gain > MAX_GAIN) gain = MAX_GAIN;

  // C. 套用 Gain + Clip + Hanning
  for (int i = 0; i < WINDOW_SIZE; i++) {
    float val = vReal[i] * gain;
    if (val > 1.0f)  val = 1.0f;
    if (val < -1.0f) val = -1.0f;
    vReal[i] = val * hanning_window[i];
    vImag[i] = 0.0f;
  }

  // D. FFT
  FFT.compute(vReal, vImag, WINDOW_SIZE, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, WINDOW_SIZE);
  int spectrum_len = WINDOW_SIZE / 2 + 1;

  // E. Binning 成 64 維
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
    for (int j = start; j < end; j++) {
      sum += vReal[j];
    }
    output_features[i] = sum / (end - start);
  }

  // F. EQ（低頻壓一點）
  output_features[0] *= 0.5f;
  output_features[1] *= 0.75f;

  // G. log1p
  for (int i = 0; i < N_FFT_BINS; i++) {
    output_features[i] = logf(output_features[i] + 1.0f);
  }
}

// ================= 輸出 FEAT =================
void output_feat_line(const float* feat) {
  // 這裡 label 固定 -1，之後用 Python 再標
  Serial.print("FEAT:-1");
  for (int i = 0; i < N_FFT_BINS; i++) {
    Serial.print(",");
    Serial.print(feat[i], 6);   // 6 位小數，方便後處理
  }
  Serial.println();
}

// ================= Setup =================
void setup() {
  Serial.begin(115200);

  ring_buffer     = (int16_t*) malloc(WINDOW_SIZE * sizeof(int16_t));
  hanning_window  = (float*)   malloc(WINDOW_SIZE * sizeof(float));
  vReal           = (float*)   malloc(WINDOW_SIZE * sizeof(float));
  vImag           = (float*)   malloc(WINDOW_SIZE * sizeof(float));

  if (!ring_buffer || !hanning_window || !vReal || !vImag) {
    Serial.println("❌ Malloc Failed!");
    while (1) { delay(1000); }
  }

  precompute_hanning();
  i2s_install();

  Serial.println("🚀 ESP32 KWS Recorder Ready. (輸出 FEAT:-1, f0..f63)");
}

// ================= Loop =================
void loop() {
  const size_t i2s_read_len = 1024;
  char   i2s_read_buff[i2s_read_len];
  size_t bytes_read;

  i2s_read(I2S_PORT, (void*)i2s_read_buff, i2s_read_len, &bytes_read, portMAX_DELAY);

  int16_t* samples = (int16_t*) i2s_read_buff;
  int count = bytes_read / 2;

  for (int i = 0; i < count; i++) {
    ring_buffer[ring_head] = samples[i];
    ring_head = (ring_head + 1) % WINDOW_SIZE;
    samples_newly_added++;
  }

  if (samples_newly_added >= STRIDE_SIZE) {
    samples_newly_added = 0;

    float features[N_FFT_BINS];
    extract_features(features);

    // 音量太小就略過，不輸出特徵（避免一堆背景）
    if (last_rms < MIN_RMS_THRESHOLD) {
      return;
    }

    // 輸出一行 FEAT
    output_feat_line(features);
  }
}

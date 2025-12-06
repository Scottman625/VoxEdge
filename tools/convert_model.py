import tensorflow as tf
import numpy as np

# 1. 載入你的模型
model = tf.keras.models.load_model("kws_model_esp32.h5")

# 2. 轉換為 TFLite (Float32 格式，保持精度)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. 儲存為 .tflite 檔案 (可選)
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# 4. 轉換為 C++ 陣列 (Hex Dump)
def hex_to_c_array(hex_data, var_name):
    c_str = ''
    c_str += '#include <cstdint>\n'
    c_str += '#include <cstddef>\n\n' # 加入這行以定義 size_t
    c_str += f'alignas(16) const unsigned char {var_name}[] = {{\n'
    
    for i, val in enumerate(hex_data):
        c_str += f'0x{val:02x}, '
        if (i + 1) % 12 == 0:
            c_str += '\n'
            
    c_str += '\n};\n\n'
    c_str += f'const unsigned int {var_name}_len = {len(hex_data)};\n'
    return c_str

c_code = hex_to_c_array(tflite_model, "g_model")

with open("model_data.h", "w") as f:
    f.write(c_code)

print("✅ 轉換完成！請將 'model_data.h' 檔案複製到你的 Arduino 專案資料夾中。")

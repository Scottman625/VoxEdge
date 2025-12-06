# export_tflite.py
import tensorflow as tf

keras_model_path = "kws_model_esp32.h5"
tflite_path = "kws_model_esp32.tflite"

model = tf.keras.models.load_model(keras_model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# 這裡先用 float32，跟你 ESP32 目前 float 模型相同
converter.optimizations = []  # 之後要量化再調
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("Saved TFLite to", tflite_path)

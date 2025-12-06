import tensorflow as tf
m = tf.keras.models.load_model("kws_model.h5")
print("model.output_shape:", m.output_shape)
print("labels_count: 5 (kai,qi,guan,bi,silence)")

import soundfile as sf, numpy as np, glob
files = glob.glob('dataset_final_train\\kai\\*.wav')[:10]
for f in files:
    d, sr = sf.read(f)
    print(f, "sec:", len(d)/sr, "rms:", np.sqrt(np.mean(d**2)))
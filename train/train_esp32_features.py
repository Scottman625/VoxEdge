import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

N_FFT_BINS = 64

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="dataset_esp32/esp32_kws_dataset_merged.npz",
        help="合併好的 npz (內含 X, y)"
    )
    parser.add_argument(
        "--out",
        default="kws_model_esp32.h5",
        help="輸出模型路徑"
    )
    args = parser.parse_args()

    # 標籤順序要跟 ESP32 LABELS 一致
    labels = ["kai", "qi", "guan", "bi", "silence"]

    if not os.path.isfile(args.data):
        print(f"❌ Data file not found: {args.data}")
        return

    print("📂 Loading dataset from:", args.data)
    data = np.load(args.data)
    X = data["X"].astype(np.float32)      # (N, 64)
    y_indices = data["y"].astype(np.int64)  # (N,)

    if X.ndim != 2 or X.shape[1] != N_FFT_BINS:
        print(f"❌ X shape expected (N,{N_FFT_BINS}), got {X.shape}")
        return

    if X.shape[0] == 0:
        print("❌ Error: Empty dataset.")
        return

    # groups：這裡簡單用 index 字串當 group id
    groups = np.array([f"g{i}" for i in range(X.shape[0])])

    print(f"📊 Dataset built. Shape: {X.shape}")
    print("   Label distribution (label_index, count):", np.unique(y_indices, return_counts=True))

    # 2. Split by group
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_indices, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_indices[train_idx], y_indices[test_idx]

    print("Train:", X_train.shape, "Test:", X_test.shape)

    # 3. Class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("⚖️ Class Weights:", class_weight_dict)

    # 4. 模型
    feature_dim = X.shape[1]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(feature_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(labels), activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("🚀 Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=32,
        class_weight=class_weight_dict,
    )

    # 5. 評估並存檔
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"📈 Test accuracy = {acc:.3f}, loss = {loss:.3f}")

    model.save(args.out)
    print(f"✅ Model saved to {args.out}")

if __name__ == '__main__':
    main()

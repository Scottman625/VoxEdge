# test_esp32_model.py
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

LABELS = ["kai", "qi", "guan", "bi", "silence"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="dataset_esp32/esp32_kws_dataset_merged.npz",
        help="包含 X, y 的 npz 檔"
    )
    parser.add_argument(
        "--model",
        default="kws_model_esp32.h5",
        help="訓練好的 Keras 模型"
    )
    args = parser.parse_args()

    print("📂 Loading dataset from:", args.data)
    data = np.load(args.data)
    X = data["X"].astype(np.float32)  # (N, 64)
    y_true = data["y"].astype(np.int64)

    print("📂 Loading model from:", args.model)
    model = tf.keras.models.load_model(args.model)

    print("🚀 Running inference on dataset...")
    y_prob = model.predict(X, batch_size=64, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    acc = (y_pred == y_true).mean()
    print(f"\n📈 Overall accuracy: {acc:.4f}")

    # 混淆矩陣
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    print("\n🔢 Confusion Matrix (rows=true, cols=pred):")
    print("      " + "  ".join(f"{lbl[:3]:>3}" for lbl in LABELS))
    for i, row in enumerate(cm):
        row_str = " ".join(f"{n:3d}" for n in row)
        print(f"{LABELS[i][:3]:>3} | {row_str}")

    # 詳細報告
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABELS))

if __name__ == "__main__":
    main()

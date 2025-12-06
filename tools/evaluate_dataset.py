"""
Evaluate model on a labeled dataset (folder per label).

Usage:
  python .\tools\evaluate_dataset.py --data dataset_synthetic_expanded --model kws_model_finetuned.h5 --out diagnostics.csv

Options:
  --labels optionally provide comma-separated label order to use (e.g. on,off,silence)

Outputs:
  - CSV with columns: file,label,predicted,score
  - Prints classification report and confusion matrix
"""
import os
import glob
import argparse
import numpy as np
import librosa
import csv
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Feature params (must match training)
SAMPLE_RATE = 16000
DURATION = 0.75
N_FFT_BINS = 64


def extract_fft_features_from_wave(y):
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    window = np.hanning(len(y))
    spectrum = np.fft.rfft(window * y)
    mag = np.abs(spectrum)
    if len(mag) < N_FFT_BINS:
        mag = np.pad(mag, (0, N_FFT_BINS - len(mag)))
    else:
        mag = mag[:N_FFT_BINS]
    feat = np.log1p(mag)
    return feat.astype(np.float32)


def load_files(data_dir, labels=None):
    files = []
    for lbl in labels:
        pattern = os.path.join(data_dir, lbl, "*.wav")
        files.extend(glob.glob(pattern))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dataset folder (subfolders per label)")
    parser.add_argument("--model", required=True, help="model path (.h5)")
    parser.add_argument("--out", default="diagnostics.csv", help="CSV output path")
    parser.add_argument("--labels", help="comma-separated label order (overrides folder order)")
    args = parser.parse_args()

    # determine labels
    if args.labels:
        labels = [l.strip() for l in args.labels.split(',')]
    else:
        # stable sort of subfolders
        labels = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    print("Using labels order:", labels)

    # load model
    model = tf.keras.models.load_model(args.model)
    print("Loaded model. Input shape:", model.input_shape)

    files = load_files(args.data, labels)
    if not files:
        print("No files found for provided labels/data")
        return
    print(f"Found {len(files)} files. Running predictions...")

    rows = []
    y_true = []
    y_pred = []

    # determine feature dimension and model input dimension
    # generate one sample to get feature length
    sample_f = None
    for ftmp in files:
        sample_f = ftmp
        break
    sample_y, _ = librosa.load(sample_f, sr=SAMPLE_RATE, duration=DURATION)
    sample_feat = extract_fft_features_from_wave(sample_y)
    feat_len = sample_feat.shape[0]
    model_input_len = model.input_shape[1] if model.input_shape and len(model.input_shape) > 1 else None
    if model_input_len is None:
        print("Warning: cannot determine model input length; proceeding with feature length")
        model_input_len = feat_len
    if model_input_len != feat_len:
        print(f"Warning: model expects input dim {model_input_len} but features have dim {feat_len}. Will pad/trim features to match model.")

    for f in files:
        try:
            y, _ = librosa.load(f, sr=SAMPLE_RATE, duration=DURATION)
        except Exception as e:
            print("Failed to load", f, e)
            continue
        feat = extract_fft_features_from_wave(y)
        # pad or trim to match model input
        if feat.shape[0] < model_input_len:
            feat = np.pad(feat, (0, model_input_len - feat.shape[0]))
        elif feat.shape[0] > model_input_len:
            feat = feat[:model_input_len]
        inp = feat.reshape(1, -1)
        pred = model.predict(inp, verbose=0)[0]
        idx = int(np.argmax(pred))
        score = float(pred[idx])
        pred_label = labels[idx] if idx < len(labels) else str(idx)
        true_label = os.path.basename(os.path.dirname(f))
        rows.append((f, true_label, pred_label, score))
        y_true.append(true_label)
        y_pred.append(pred_label)

    # write CSV
    with open(args.out, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["file", "label", "predicted", "score"])
        writer.writerows(rows)
    print(f"Wrote per-file results to {args.out}")

    # report
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels, target_names=labels))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))

if __name__ == '__main__':
    main()

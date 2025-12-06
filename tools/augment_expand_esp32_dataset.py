import numpy as np

def augment_silence_features(X_silence, num_aug=3):
    """
    X_silence: (N, 64) 來自 ESP32 的 silence 特徵
    num_aug:   每筆額外產生幾個變體
    回傳: 擴增後的新矩陣 (N * (1+num_aug), 64)
    """
    all_feats = [X_silence]
    N, D = X_silence.shape

    for _ in range(num_aug):
        # 隨機加一點小 noise（feature 空間）
        noise = np.random.normal(loc=0.0, scale=0.01, size=(N, D))
        # 隨機縮放：模擬不同 SNR/增益
        scale = np.random.uniform(0.9, 1.1, size=(N, 1))
        aug = X_silence * scale + noise
        all_feats.append(aug)

    return np.vstack(all_feats)

# 載入各類
data_qi = np.load("esp32_kws_dataset_qi.npz")
data_bi = np.load("esp32_kws_dataset_bi.npz")
data_kai = np.load("esp32_kws_dataset_kai.npz")
data_guan = np.load("esp32_kws_dataset_guan.npz")
data_sil = np.load("esp32_kws_dataset_silence.npz")

X = np.vstack([data_kai["X"], data_qi["X"], data_guan["X"], data_bi["X"], data_sil["X"]])
y = np.concatenate([data_kai["y"], data_qi["y"], data_guan["y"], data_bi["y"], data_sil["y"]])

# 對 silence 做特徵級擴增
sil_mask = (y == 4)        # 假設 4 是 silence
X_sil = X[sil_mask]
X_sil_aug = augment_silence_features(X_sil, num_aug=3)
y_sil_aug = np.full(X_sil_aug.shape[0], 4, dtype=np.int64)

# 把擴增的 silence 接回去
X_all = np.vstack([X[~sil_mask], X_sil_aug])
y_all = np.concatenate([y[~sil_mask], y_sil_aug])

np.savez("esp32_kws_dataset_merged.npz", X=X_all, y=y_all)
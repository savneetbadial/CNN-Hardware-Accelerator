import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA, PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. AAMI MAPPING & SIGNAL PREPROCESSING


AAMI_MAPPING = {
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,  # Class N: Normal
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,  # Class S: Supraventricular Ectopic (SVEB)
    "V": 2,
    "E": 2,  # Class V: Ventricular Ectopic (VEB)
    "F": 3,  # Class F: Fusion Beat
    "/": 4,
    "f": 4,
    "Q": 4,  # Class Q: Unknown
}

CLASS_NAMES = [
    "N (Normal)",
    "S (SVEB)",
    "V (VEB)",
    "F (Fusion)",
    "Q (Unknown)",
]


def preprocess_signal(
    signal, lowcut=0.5, highcut=40.0, fs=360.0, order=4
):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)



# 2. LOAD MIT-BIH CSV DATA


def load_local_mitbih_csv(data_dir=".", window_size=180, fs=360.0):

    ekg_files = sorted(glob.glob(os.path.join(data_dir, "*_ekg.csv")))

    if not ekg_files:
        raise FileNotFoundError(
            f"No *_ekg.csv files found in directory '{data_dir}'!"
        )

    half_win = window_size // 2
    heartbeats = []
    labels = []

    print(
        f"Found {len(ekg_files)} local ECG CSV files. Processing heartbeat windows..."
    )

    for ekg_path in ekg_files:
        rec_id = (
            os.path.basename(ekg_path).replace("_ekg.csv", "").split(".")[0]
        )
        ann_path = os.path.join(data_dir, f"{rec_id}_annotations_1.csv")

        if not os.path.exists(ann_path):
            print(
                f"Skipping record {rec_id}: missing annotation file ({ann_path})"
            )
            continue

        # Load EKG signal
        df_ekg = pd.read_csv(ekg_path)
        # Handle single column or multi-column EKG CSVs
        if df_ekg.shape[1] > 1:
            ecg_signal = df_ekg.iloc[:, 1].values  # Select signal column
        else:
            ecg_signal = df_ekg.iloc[:, 0].values

        # Apply Butterworth filtering
        filtered_ecg = preprocess_signal(ecg_signal, fs=fs)

        # Load Annotations
        df_ann = pd.read_csv(ann_path)

        # Detect column names for sample index and symbol label
        sample_col = [
            c
            for c in df_ann.columns
            if "sample" in c.lower() or "idx" in c.lower() or "index" in c.lower()
        ]
        symbol_col = [
            c
            for c in df_ann.columns
            if "symbol" in c.lower()
            or "type" in c.lower()
            or "annotation" in c.lower()
        ]

        s_col = sample_col[0] if sample_col else df_ann.columns[0]
        lbl_col = symbol_col[0] if symbol_col else df_ann.columns[1]

        for _, row in df_ann.iterrows():
            sample_idx = int(row[s_col])
            symbol = str(row[lbl_col]).strip()

            if symbol in AAMI_MAPPING:
                if (
                    sample_idx - half_win >= 0
                    and sample_idx + half_win < len(filtered_ecg)
                ):
                    beat = filtered_ecg[
                        sample_idx - half_win : sample_idx + half_win
                    ]
                    heartbeats.append(beat)
                    labels.append(AAMI_MAPPING[symbol])

    X_signals = np.array(heartbeats)
    y_labels = np.array(labels)

    print(
        f"\nExtracted Total Beat Windows: {X_signals.shape[0]} samples of length {X_signals.shape[1]}"
    )
    for c_idx, c_name in enumerate(CLASS_NAMES):
        count = np.sum(y_labels == c_idx)
        print(f"  Class {c_idx} [{c_name}]: {count} beats")

    return X_signals, y_labels


def extract_handcrafted_features(
    X_raw, n_components_pca=10, n_components_ica=5
):
    print(
        f"\nExtracting features using PCA ({n_components_pca}) & FastICA ({n_components_ica})..."
    )
    pca = PCA(n_components=n_components_pca, random_state=42)
    ica = FastICA(n_components=n_components_ica, random_state=42, max_iter=1000)

    X_pca = pca.fit_transform(X_raw)
    X_ica = ica.fit_transform(X_raw)

    return np.hstack((X_pca, X_ica))


# 3. HYBRID CNN-TRANSFORMER MODEL

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_dual_branch_model(
    signal_shape=(180, 1), feature_shape=(15,), num_classes=5
):
    # Branch 1: CNN + Transformer
    signal_input = layers.Input(shape=signal_shape, name="ecg_signal_input")

    x = layers.Conv1D(filters=32, kernel_size=5, padding="same")(signal_input)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    x = transformer_encoder(
        x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1
    )
    x = layers.GlobalAveragePooling1D()(x)

    # Branch 2: PCA + FastICA branch
    feature_input = layers.Input(
        shape=feature_shape, name="handcrafted_feature_input"
    )
    y = layers.Dense(32, activation="relu")(feature_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)

    # Merge branches
    merged = layers.concatenate([x, y])
    z = layers.Dense(64, activation="relu")(merged)
    z = layers.Dropout(0.3)(z)

    # 5-Class Output Head
    output = layers.Dense(
        num_classes, activation="softmax", name="classification_head"
    )(z)

    model = models.Model(
        inputs=[signal_input, feature_input],
        outputs=output,
        name="ECG_CNN_Transformer",
    )
    return model


# 4. CROSS-VALIDATION TRAINING & EVALUATION PIPELINE

def print_multiclass_metrics(y_true, y_pred):
    print("\n================ Aggregated Classification Report ================")
    print(
        classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, digits=4
        )
    )


def plot_5x5_confusion_matrix(
    y_true, y_pred, save_path="confusion_matrix_5class.png"
):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("5-Class Normalized Confusion Matrix (MIT-BIH AAMI EC57)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def run_cross_validation(X_signal, X_features, y_labels, epochs=25, batch_size=64):
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_signal, y_labels), start=1
    ):
        print(f"\n================ Training Fold {fold}/{N_SPLITS} ================")

        X_sig_train, X_sig_val = X_signal[train_idx], X_signal[val_idx]
        X_feat_train, X_feat_val = X_features[train_idx], X_features[val_idx]
        y_train, y_val = y_labels[train_idx], y_labels[val_idx]

        # Compute balanced class weights to handle 5-class AAMI imbalance
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))

        model = build_dual_branch_model(
            signal_shape=(X_signal.shape[1], 1),
            feature_shape=(X_features.shape[1],),
            num_classes=5,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2
            ),
        ]

        model.fit(
            x=[X_sig_train, X_feat_train],
            y=y_train,
            validation_data=([X_sig_val, X_feat_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1,
        )

        val_preds = model.predict([X_sig_val, X_feat_val])
        val_pred_labels = np.argmax(val_preds, axis=1)

        all_y_true.extend(y_val)
        all_y_pred.extend(val_pred_labels)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    print_multiclass_metrics(all_y_true, all_y_pred)
    plot_5x5_confusion_matrix(all_y_true, all_y_pred)


if __name__ == "__main__":
    # 1. Load heartbeat windows directly from your uploaded CSV files in current folder '.'
    X_signals_raw, y_labels = load_local_mitbih_csv(data_dir=".")

    # 2. Reshape signals for 1D CNN input
    X_signals = np.expand_dims(X_signals_raw, axis=-1)

    # 3. Extract PCA + FastICA feature matrix
    X_features = extract_handcrafted_features(
        X_signals_raw, n_components_pca=10, n_components_ica=5
    )

    # 4. Execute 5-fold cross-validation
    run_cross_validation(X_signals, X_features, y_labels, epochs=25, batch_size=64)
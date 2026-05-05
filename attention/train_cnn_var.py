import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import os

# 1. 設定資料庫與模型儲存路徑 (對應 ASCAD Variable Key)
DATABASE_FILE = "ASCAD_data/ASCAD_databases/ascad-variable.h5"
MODEL_SAVE_PATH = "ASCAD_data/ASCAD_trained_models/cnn_var_model.h5"

def load_ascad_var(database_file):
    print(f"[*] 正在載入變動金鑰資料庫: {database_file} ...")
    try:
        in_file = h5py.File(database_file, "r")
    except Exception as e:
        print(f"[!] 載入失敗: {e}")
        exit()
        
    # Variable Key 的資料量非常大 (訓練集高達 20 萬條，特徵點 1400)
    print("[*] 正在將 20 萬條訓練波形載入記憶體 (這可能需要幾十秒)...")
    X_profiling = np.array(in_file['Profiling_traces']['traces'], dtype=np.float32)
    Y_profiling = to_categorical(in_file['Profiling_traces']['labels'], num_classes=256)
    
    print("[*] 正在將 10 萬條測試波形載入記憶體...")
    X_attack = np.array(in_file['Attack_traces']['traces'], dtype=np.float32)
    Y_attack = to_categorical(in_file['Attack_traces']['labels'], num_classes=256)
    
    # Reshape 給 1D-CNN 吃 (樣本數, 特徵長度, 1)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    
    print(f"[*] 訓練集維度: {X_profiling.shape}, 標籤維度: {Y_profiling.shape}")
    print(f"[*] 測試集維度: {X_attack.shape}, 標籤維度: {Y_attack.shape}")
    
    return X_profiling, Y_profiling, X_attack, Y_attack

def build_cnn_var_model(input_length):
    print("[*] 正在建構 Variable Key 專用強化版 CNN 架構...")
    model = Sequential(name="CNN_Variable_Key")
    
    # 輸入層 (自動抓取 Variable Key 的 1400 個特徵點)
    model.add(Input(shape=(input_length, 1)))
    
    # --- Block 1: 抓取局部特徵 ---
    # 增加 filters 數量來應對更高維度的變動特徵
    model.add(Conv1D(filters=32, kernel_size=11, activation='selu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2, strides=2))
    
    # --- Block 2: 抓取中距離特徵 ---
    model.add(Conv1D(filters=64, kernel_size=25, activation='selu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2, strides=2))
    
    # --- Block 3: 加深網路以萃取跨越遮罩的全域關聯特徵 ---
    model.add(Conv1D(filters=128, kernel_size=51, activation='selu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=4, strides=4))
    
    # --- Classification Block ---
    model.add(Flatten())
    
    # 加大 Dense 層容量，讓它有足夠的「腦容量」記住 20 萬把變動金鑰的複雜邏輯
    model.add(Dense(128, activation='selu'))
    model.add(Dense(128, activation='selu'))
    
    model.add(Dense(256, activation='softmax'))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    model.summary()
    return model

if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    X_train, Y_train, X_val, Y_val = load_ascad_var(DATABASE_FILE)
    
    input_length = X_train.shape[1]
    model = build_cnn_var_model(input_length)
    
    print("\n[*] ====== 開始訓練 (Variable Key Phase) ======")
    # Variable key 需要更多 Epoch 才能收斂，我們設定 75 次
    history = model.fit(
        X_train, Y_train,
        batch_size=200,  # 稍微調大 batch_size，把雙 L40S 的吞吐量榨乾
        epochs=75,
        validation_data=(X_val, Y_val),
        verbose=1
    )
    
    model.save(MODEL_SAVE_PATH)
    print(f"\n[*] 訓練完成！模型已儲存至: {MODEL_SAVE_PATH}")

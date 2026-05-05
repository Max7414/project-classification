import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import os

# 1. 設定資料庫路徑 (對應 ASCAD fixed key 資料集)
DATABASE_FILE = "ASCAD_data/ASCAD_databases/ASCAD.h5"
MODEL_SAVE_PATH = "ASCAD_data/ASCAD_trained_models/cnnd_paper_model.h5"

def load_ascad(database_file):
    print(f"[*] 正在載入資料庫: {database_file} ...")
    try:
        in_file = h5py.File(database_file, "r")
    except Exception as e:
        print(f"[!] 載入失敗: {e}")
        exit()
        
    # 讀取 Profiling (訓練集) 的波形與標籤 (S-box 輸出)
    X_profiling = np.array(in_file['Profiling_traces']['traces'], dtype=np.float32)
    # 將標籤轉為 One-Hot Encoding (256 類別)
    Y_profiling = to_categorical(in_file['Profiling_traces']['labels'], num_classes=256)
    
    # 讀取 Attack (驗證集) 的波形與標籤
    X_attack = np.array(in_file['Attack_traces']['traces'], dtype=np.float32)
    Y_attack = to_categorical(in_file['Attack_traces']['labels'], num_classes=256)
    
    # 將波形 Reshape 為 Keras 1D-CNN 所需的維度: (樣本數, 長度, 通道數)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    
    print(f"[*] 訓練集維度: {X_profiling.shape}, 標籤維度: {Y_profiling.shape}")
    print(f"[*] 測試集維度: {X_attack.shape}, 標籤維度: {Y_attack.shape}")
    
    return X_profiling, Y_profiling, X_attack, Y_attack

def build_cnnd_model(input_length):
    print("[*] 正在建構論文提出的 CNNd 架構...")
    model = Sequential(name="CNNd_Model")
    
    # 輸入層 (ASCAD v1 fixed key 通常是 700 個 POI)
    model.add(Input(shape=(input_length, 1)))
    
    # --- CONV 1 Block ---
    # kernels: 4, size: 3, activation: SeLU, initialization: He Uniform
    model.add(Conv1D(filters=4, kernel_size=3, activation='selu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2, strides=2))
    
    # --- CONV 2 Block ---
    # kernels: 8, size: 51, activation: SeLU, initialization: He Uniform
    model.add(Conv1D(filters=8, kernel_size=51, activation='selu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2, strides=2))
    
    # --- Classification Block ---
    model.add(Flatten())
    
    # 2層 Fully Connected，每層 10 個神經元
    model.add(Dense(10, activation='selu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='selu', kernel_initializer='he_uniform'))
    
    # 輸出層: 256 個神經元，Softmax
    model.add(Dense(256, activation='softmax'))
    
    # 編譯模型
    # 論文提到使用 Adam 與 MSE，但實務上針對 256 類別的多分類問題，
    # 使用 Categorical Crossentropy 在收斂上更為合理且標準。此處我們採用標準配置。
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    model.summary()
    return model

if __name__ == "__main__":
    # 確保儲存目錄存在
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # 1. 載入資料
    X_train, Y_train, X_val, Y_val = load_ascad(DATABASE_FILE)
    
    # 2. 建構模型
    input_length = X_train.shape[1]
    model = build_cnnd_model(input_length)
    
    # 3. 開始訓練 (參數: batch_size=50, epochs=50)
    print("\n[*] ====== 開始訓練 (Training Phase) ======")
    history = model.fit(
        X_train, Y_train,
        batch_size=50,
        epochs=50,
        validation_data=(X_val, Y_val),
        verbose=1
    )
    
    # 4. 儲存模型
    model.save(MODEL_SAVE_PATH)
    print(f"\n[*] 訓練完成！模型已成功儲存至: {MODEL_SAVE_PATH}")
    print("[*] 您現在可以使用 test_models.py 將此模型掛載並繪製 Key Rank 圖表了！")

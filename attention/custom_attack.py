import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, AveragePooling1D, 
                                     Dense, BatchNormalization, LSTM, 
                                     Multiply, Lambda, Softmax, Concatenate)
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# ==========================================
# 1. 模型架構定義區 (直接移植你的訓練架構)
# ==========================================
def build_attention_model(input_length):
    print(f"[*] 正在建構空殼模型 (Input Length: {input_length})...")
    
    inputs = Input(shape=(input_length, 1))
    
    # 1. Junior Encoder
    x = Conv1D(filters=32, kernel_size=11, strides=5, activation='selu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    
    # 2. Senior Encoder
    fw_lstm = LSTM(128, return_sequences=True, kernel_constraint=tf.keras.constraints.UnitNorm())(x)
    bw_lstm = LSTM(128, return_sequences=True, go_backwards=True, kernel_constraint=tf.keras.constraints.UnitNorm())(x)
    
    # 3. Attention Mechanism
    def build_attention_block(lstm_output, name_prefix):
        score = Conv1D(filters=1, kernel_size=1, use_bias=False, name=f'{name_prefix}_score')(lstm_output)
        score = BatchNormalization(name=f'{name_prefix}_bn')(score)
        att_weights = Softmax(axis=1, name=f'{name_prefix}_softmax')(score)
        weighted_seq = Multiply()([lstm_output, att_weights])
        context_vector = Lambda(lambda t: tf.reduce_sum(t, axis=1), name=f'{name_prefix}_context')(weighted_seq)
        return context_vector

    fw_context = build_attention_block(fw_lstm, "fw_att")
    bw_context = build_attention_block(bw_lstm, "bw_att")
    
    # 4. Classifier
    merged = Concatenate()([fw_context, bw_context])
    dense = Dense(128, activation='selu')(merged)
    outputs = Dense(256, activation='softmax')(dense)
    
    model = Model(inputs=inputs, outputs=outputs, name="Attention_Variant2")
    return model

# ==========================================
# 2. 基本參數與 AES S-box
# ==========================================
DB_FILE = "ASCAD_data/ASCAD_databases/ascad-variable.h5"
WEIGHTS_FILE = "ASCAD_data/ASCAD_trained_models/attention_var_model.h5"
NUM_TRACES = 2000
TARGET_BYTE = 2  # ASCAD 預設攻擊第 3 個 Byte

SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

if __name__ == "__main__":
    # ==========================================
    # 3. 載入波形與資料
    # ==========================================
    print(f"[*] 正在載入 ASCAD 資料庫: {DB_FILE} ...")
    try:
        in_file = h5py.File(DB_FILE, "r")
    except Exception as e:
        print("[!] 資料庫載入失敗:", e)
        sys.exit(-1)
        
    X_attack = in_file['Attack_traces']['traces'][:NUM_TRACES]
    # 自動抓取輸入長度 (通常是 1400 或 700)
    input_length = X_attack.shape[1] 
    
    # Reshape 讓神經網路吃得下 (Batch, Length, Channels)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    
    plaintexts = in_file['Attack_traces']['metadata']['plaintext'][:NUM_TRACES, TARGET_BYTE]
    keys = in_file['Attack_traces']['metadata']['key'][:NUM_TRACES, TARGET_BYTE]
    
    TRUE_KEY = keys[0]
    print(f"[*] 載入完成！目標真實金鑰為: 0x{TRUE_KEY:02X}")

    # ==========================================
    # 4. 模型重建與權重注入 (The Magic Trick)
    # ==========================================
    try:
        # 1. 建立空殼
        model = build_attention_model(input_length)
        # 2. 注入靈魂 (權重)
        model.load_weights(WEIGHTS_FILE)
        print("[*] 權重注入成功！完全避開了 Lambda 載入的噩夢！")
    except Exception as e:
        print("[!] 權重注入失敗，請確認你的檔案路徑是否正確。")
        print("錯誤訊息:", e)
        sys.exit(-1)

    # ==========================================
    # 5. AI 推論與破解計算 (Guessing Entropy)
    # ==========================================
    print(f"[*] 開始讓 AI 推論 {NUM_TRACES} 條波形...")
    predictions = model.predict(X_attack)

    print("[*] 開始進行側信道對數似然計算 (Log-Likelihood)...")
    key_log_probs = np.zeros(256)
    guessing_entropy = []

    for i in tqdm(range(NUM_TRACES), desc="攻擊進度"):
        p = plaintexts[i]
        pred_probs = predictions[i]
        
        # 暴力猜測 256 種金鑰
        for k_guess in range(256):
            hyp_sbox_out = SBOX[p ^ k_guess]
            # 加上 1e-36 避免 log(0)
            key_log_probs[k_guess] += np.log(pred_probs[hyp_sbox_out] + 1e-36)
        
        ranked_keys = np.argsort(key_log_probs)[::-1]
        rank = np.where(ranked_keys == TRUE_KEY)[0][0]
        guessing_entropy.append(rank)

    # ==========================================
    # 6. 輸出火力展示圖
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(guessing_entropy, color='red', linewidth=2, label="Attention Model Attack")
    plt.title(f'Custom SCA Attack on Variable Key (Traces: {NUM_TRACES})')
    plt.xlabel('Number of Traces')
    plt.ylabel('Guessing Entropy (Rank of True Key)')
    plt.legend()
    plt.grid(True)
    
    save_path = "custom_attack_result.png"
    plt.savefig(save_path)
    
    print(f"\n[*] 攻擊結束！最終金鑰排名: {guessing_entropy[-1]} (只要是 0 就是破關！)")
    print(f"[*] 破解曲線圖已儲存至: {save_path}")

"""
=============================================================================
第三章：用 nn.Module 建構神經網路
=============================================================================

從手動到自動：
─────────────
前兩章我們手動管理 weight 和 bias，手動計算前向傳播。
但真實的神經網路有數百萬個參數，不可能手動管理。

PyTorch 提供了 nn.Module 這個強大的基礎類別：
- 自動管理所有參數
- 自動建立計算圖
- 提供豐富的現成層（Linear, Conv2d, LSTM...）
- 支援模型的儲存與載入

什麼是神經網路？（白話文版）
─────────────────────────
想像一個工廠的流水線：

  原料 → [加工站1] → [加工站2] → [加工站3] → 成品
  輸入     Layer 1     Layer 2     Layer 3    輸出

每個加工站（Layer）做兩件事：
1. 線性轉換：output = input × weight + bias
2. 非線性激活：output = activation_function(output)

為什麼需要非線性激活？
→ 如果只有線性轉換，不管多少層，最終結果還是線性的
→ 加了激活函數後，神經網路才能學到複雜的非線性關係

本章學習目標：
─────────────
✓ 理解 nn.Module 的結構
✓ 常用的層：Linear, Conv2d, BatchNorm, Dropout
✓ 常用的激活函數：ReLU, Sigmoid, Tanh, Softmax
✓ 建構第一個完整的神經網路
✓ 實際案例：分類手寫數字（MNIST）
=============================================================================
"""

import torch
import torch.nn as nn      # 神經網路模組
import torch.nn.functional as F  # 函式版的層和激活函數

print("=" * 60)
print("第三章：用 nn.Module 建構神經網路")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 3.1 nn.Linear：全連接層（最基本的層）
# ─────────────────────────────────────────────────────────────
print("\n📌 3.1 nn.Linear：全連接層")
print("-" * 40)

# nn.Linear(in_features, out_features) 做的事：
# output = input @ weight.T + bias
#
# 圖解（in=3, out=2）：
#
#  input[0] ──w00──┐
#  input[1] ──w10──┼──→ output[0] = w00*in0 + w10*in1 + w20*in2 + bias0
#  input[2] ──w20──┘
#
#  input[0] ──w01──┐
#  input[1] ──w11──┼──→ output[1] = w01*in0 + w11*in1 + w21*in2 + bias1
#  input[2] ──w21──┘

linear = nn.Linear(in_features=3, out_features=2)

# 查看自動建立的參數
print(f"weight 形狀: {linear.weight.shape}")   # (2, 3)
print(f"bias 形狀: {linear.bias.shape}")        # (2,)
print(f"weight:\n{linear.weight}")
print(f"bias: {linear.bias}")

# 輸入一個 batch 的資料
# batch_size=4, features=3
input_data = torch.rand(4, 3)
output = linear(input_data)     # 等同於 linear.forward(input_data)
print(f"\n輸入形狀: {input_data.shape}")   # (4, 3)
print(f"輸出形狀: {output.shape}")         # (4, 2)


# ─────────────────────────────────────────────────────────────
# 3.2 激活函數
# ─────────────────────────────────────────────────────────────
print("\n\n📌 3.2 激活函數")
print("-" * 40)

x = torch.linspace(-5, 5, 11)
print(f"x = {x.tolist()}")

# ReLU：max(0, x) — 最常用！
# 優點：計算簡單、不會梯度消失
# 圖形：      ╱
#        ────╱
#            0
relu_out = F.relu(x)
print(f"\nReLU(x) = {relu_out.tolist()}")

# Sigmoid：1 / (1 + e^(-x)) — 輸出 (0, 1)
# 用途：二元分類的輸出層、機率輸出
# 圖形：    ──────
#          ╱
#    ──────
sigmoid_out = torch.sigmoid(x)
print(f"Sigmoid(x) = {[f'{v:.3f}' for v in sigmoid_out.tolist()]}")

# Tanh：(e^x - e^(-x)) / (e^x + e^(-x)) — 輸出 (-1, 1)
# 用途：RNN 中常用
tanh_out = torch.tanh(x)
print(f"Tanh(x) = {[f'{v:.3f}' for v in tanh_out.tolist()]}")

# Softmax：把任意數值轉成機率分佈（總和為 1）
# 用途：多類別分類的輸出層
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(f"\nSoftmax([2.0, 1.0, 0.1]) = {probs.tolist()}")
print(f"  總和 = {probs.sum().item():.4f}")  # 1.0

# LeakyReLU：負值不是 0，而是 0.01x
# 解決 ReLU 的「dying neurons」問題
leaky_out = F.leaky_relu(x, negative_slope=0.01)
print(f"\nLeakyReLU(x) = {[f'{v:.3f}' for v in leaky_out.tolist()]}")

print("""
  激活函數選擇指南：
  ┌─────────────┬──────────────────────────┐
  │ 隱藏層       │ ReLU（首選）              │
  │             │ LeakyReLU（避免 dead neuron）│
  ├─────────────┼──────────────────────────┤
  │ 二元分類輸出  │ Sigmoid                  │
  │ 多類別分類輸出│ Softmax (或在 loss 中處理) │
  │ 回歸輸出     │ 不加激活函數（直接輸出）    │
  └─────────────┴──────────────────────────┘
""")


# ─────────────────────────────────────────────────────────────
# 3.3 建構第一個神經網路
# ─────────────────────────────────────────────────────────────
print("\n📌 3.3 建構第一個神經網路")
print("-" * 40)

# 方法一：繼承 nn.Module（推薦，最有彈性）
class SimpleNet(nn.Module):
    """
    一個簡單的三層全連接神經網路

    結構圖：
    輸入(10) → Linear(10,64) → ReLU → Linear(64,32) → ReLU → Linear(32,3) → 輸出(3)

    應用場景：假設有 10 個特徵，要分成 3 個類別
    例如：用身高、體重、年齡等 10 個特徵，預測是貓/狗/鳥
    """
    def __init__(self):
        super().__init__()  # 一定要呼叫父類別的 __init__

        # 定義每一層
        self.fc1 = nn.Linear(10, 64)    # fc = fully connected（全連接）
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        """前向傳播：定義資料如何流過網路"""
        x = self.relu(self.fc1(x))   # 第一層 + 激活
        x = self.relu(self.fc2(x))   # 第二層 + 激活
        x = self.fc3(x)              # 輸出層（不加激活，留給 loss function）
        return x

# 建立模型實例
model = SimpleNet()
print(f"模型結構：\n{model}")

# 查看所有可訓練參數
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n總參數量: {total_params}")
print(f"可訓練參數量: {trainable_params}")

# 逐層查看參數
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

# 測試前向傳播
dummy_input = torch.rand(5, 10)   # batch_size=5, features=10
output = model(dummy_input)
print(f"\n輸入形狀: {dummy_input.shape}")  # (5, 10)
print(f"輸出形狀: {output.shape}")         # (5, 3)


# 方法二：nn.Sequential（簡單直覺，但彈性較低）
model_seq = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 3)
)
print(f"\nSequential 模型：\n{model_seq}")

# 兩種方法的結果等價
output_seq = model_seq(dummy_input)
print(f"輸出形狀: {output_seq.shape}")  # (5, 3)


# ─────────────────────────────────────────────────────────────
# 3.4 常用的層
# ─────────────────────────────────────────────────────────────
print("\n\n📌 3.4 常用的層")
print("-" * 40)

# ── Dropout：訓練時隨機丟棄一些神經元，防止過擬合 ──
# 像是考試時隨機遮住一些答案，迫使學生不要只背答案
dropout = nn.Dropout(p=0.5)  # 50% 的機率丟棄

x = torch.ones(1, 10)  # 全 1 的輸入

# 訓練模式
dropout.train()
print(f"Dropout 訓練模式（有些變 0）: {dropout(x)}")

# 評估模式（Dropout 會自動關閉）
dropout.eval()
print(f"Dropout 評估模式（全部保留）: {dropout(x)}")

# ── BatchNorm：批次正規化，加速訓練、穩定梯度 ──
# 對每個 batch 的資料做標準化（mean=0, std=1）
bn = nn.BatchNorm1d(num_features=5)  # 5 個特徵
x = torch.rand(32, 5) * 100  # 假設原始資料範圍很大
x_normalized = bn(x)
print(f"\nBatchNorm 前 — mean: {x.mean(dim=0)[:3].tolist()}")
print(f"BatchNorm 後 — mean: {x_normalized.mean(dim=0)[:3].tolist()}")

# ── Embedding：把離散的 ID 轉成連續的向量 ──
# 用途：把「單字」轉成「向量」（NLP 的基礎）
# 假設詞彙量 = 1000，每個字用 64 維向量表示
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)
word_ids = torch.tensor([42, 7, 256, 100])     # 4 個字的 ID
word_vectors = embedding(word_ids)
print(f"\nEmbedding: {word_ids.shape} → {word_vectors.shape}")  # (4,) → (4, 64)


# ─────────────────────────────────────────────────────────────
# 3.5 CNN 卷積神經網路基礎
# ─────────────────────────────────────────────────────────────
print("\n\n📌 3.5 CNN 卷積神經網路基礎")
print("-" * 40)

# Conv2d：2D 卷積層，專門處理圖片
#
# 卷積的直覺理解：
# 用一個小視窗（kernel）在圖片上「滑動」，每次計算一小塊區域的特徵
#
# 輸入圖片 (3x5x5)         kernel (3x3)         輸出特徵圖
# ┌─────────┐              ┌───┐
# │ * * * * * │  ×          │* *│ = 一個數值
# │ * [*]*[*] │             │* *│
# │ * [*]*[*] │             └───┘
# │ * * * * * │
# │ * * * * * │
# └─────────┘

conv = nn.Conv2d(
    in_channels=3,      # RGB 3 個顏色通道
    out_channels=16,    # 輸出 16 個特徵圖
    kernel_size=3,      # 3x3 的卷積核
    stride=1,           # 每次移動 1 格
    padding=1           # 邊緣補零，保持尺寸不變
)

# 輸入：一批圖片 (batch=2, channels=3, height=32, width=32)
images = torch.rand(2, 3, 32, 32)
features = conv(images)
print(f"Conv2d 輸入: {images.shape}")      # (2, 3, 32, 32)
print(f"Conv2d 輸出: {features.shape}")    # (2, 16, 32, 32)

# MaxPool2d：最大池化，縮小特徵圖尺寸
# 把每個 2x2 區域取最大值，尺寸變成一半
pool = nn.MaxPool2d(kernel_size=2, stride=2)
pooled = pool(features)
print(f"MaxPool 後: {pooled.shape}")        # (2, 16, 16, 16)

# 一個簡單的 CNN
class SimpleCNN(nn.Module):
    """
    簡單的 CNN 分類器

    結構：
    輸入圖片 (1,28,28) → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → 輸出(10)

    MNIST 手寫數字：28x28 灰階圖片，分成 0~9 共 10 類
    """
    def __init__(self):
        super().__init__()
        # 卷積區塊
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 28x28 → 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 14x14 → 14x14
        self.pool = nn.MaxPool2d(2, 2)                             # 尺寸減半

        # 全連接區塊
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 攤平後接全連接
        self.fc2 = nn.Linear(128, 10)            # 10 個類別
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.pool(self.relu(self.conv1(x)))   # → (batch, 16, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))   # → (batch, 32, 7, 7)
        x = x.flatten(1)                          # → (batch, 32*7*7=1568)
        x = self.relu(self.fc1(x))                # → (batch, 128)
        x = self.fc2(x)                           # → (batch, 10)
        return x

cnn = SimpleCNN()
print(f"\nCNN 模型：\n{cnn}")

# 測試
dummy_images = torch.rand(4, 1, 28, 28)  # 4 張 28x28 灰階圖
predictions = cnn(dummy_images)
print(f"\nCNN 輸入: {dummy_images.shape}")
print(f"CNN 輸出: {predictions.shape}")   # (4, 10) — 每張圖 10 個類別的分數


# ─────────────────────────────────────────────────────────────
# 3.6 模型的 train/eval 模式
# ─────────────────────────────────────────────────────────────
print("\n\n📌 3.6 模型的 train/eval 模式")
print("-" * 40)

print("""
某些層在訓練和評估時的行為不同：
- Dropout：訓練時隨機丟棄，評估時全部保留
- BatchNorm：訓練時用 batch 統計量，評估時用全局統計量

切換方式：
  model.train()  → 訓練模式（預設）
  model.eval()   → 評估模式
""")

model = SimpleNet()
print(f"預設模式: training={model.training}")

model.eval()
print(f"model.eval() 後: training={model.training}")

model.train()
print(f"model.train() 後: training={model.training}")


# ─────────────────────────────────────────────────────────────
# 3.7 模型搬到 GPU
# ─────────────────────────────────────────────────────────────
print("\n\n📌 3.7 模型搬到 GPU")
print("-" * 40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

# 把模型搬到 GPU
model = SimpleNet().to(device)

# 輸入資料也要搬到同一個裝置！
input_data = torch.rand(5, 10).to(device)
output = model(input_data)
print(f"GPU 上的運算完成，輸出形狀: {output.shape}")


# ─────────────────────────────────────────────────────────────
# 3.8 自訂複雜模型：殘差連接（ResNet 的核心）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 3.8 進階：殘差連接（Skip Connection）")
print("-" * 40)

class ResidualBlock(nn.Module):
    """
    殘差區塊：output = F(x) + x

    直覺理解：
    ─────────────────────────
    普通網路：x → [Layer] → output
    殘差網路：x → [Layer] → output + x
                  ↑                  ↑
                  └────── 捷徑 ──────┘

    好處：
    1. 解決深層網路的「梯度消失」問題
    2. 讓網路更容易學習（只需要學 F(x) = output - x 這個「殘差」）
    3. 使超深的網路（100+ 層）也能有效訓練
    """
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x                  # 保存輸入
        out = self.block(x)           # 通過兩層線性轉換
        out = out + residual          # 加上原始輸入（殘差連接！）
        out = self.relu(out)          # 最後的激活
        return out


class ResNet(nn.Module):
    """使用殘差區塊的分類網路"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # 堆疊多個殘差區塊
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x


resnet = ResNet(input_dim=20, hidden_dim=64, num_classes=5, num_blocks=3)
print(f"ResNet 模型：\n{resnet}")

dummy = torch.rand(8, 20)
out = resnet(dummy)
print(f"\n輸入: {dummy.shape} → 輸出: {out.shape}")

# 參數量
total = sum(p.numel() for p in resnet.parameters())
print(f"總參數量: {total}")


# ─────────────────────────────────────────────────────────────
# 3.9 練習題
# ─────────────────────────────────────────────────────────────
print("\n\n📌 3.9 練習題")
print("-" * 40)
print("""
練習 1：建立一個 nn.Sequential 模型：
        Linear(784, 256) → ReLU → Dropout(0.3) → Linear(256, 128)
        → ReLU → Dropout(0.3) → Linear(128, 10)
        計算總參數量

練習 2：建立一個 CNN：
        Conv2d(3,32,3,padding=1) → ReLU → MaxPool(2)
        → Conv2d(32,64,3,padding=1) → ReLU → MaxPool(2)
        → Flatten → Linear(64*8*8, 256) → ReLU → Linear(256, 10)
        輸入形狀為 (batch, 3, 32, 32)，確認輸出形狀是 (batch, 10)

練習 3：在 ResidualBlock 中加入 BatchNorm1d 層
        測試加入前後的輸出差異
""")

print("\n" + "=" * 60)
print("第三章結束！下一章：訓練工作流程（Loss, Optimizer, Training Loop）")
print("=" * 60)

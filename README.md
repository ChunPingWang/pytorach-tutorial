# 🔥 PyTorch 完整教學課程：從零開始到實戰應用

<p align="center">
  <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="200" alt="PyTorch Logo">
</p>

<p align="center">
  <strong>專為初學者設計 ｜ 大量實際案例 ｜ 中文詳細註解 ｜ 可直接執行</strong>
</p>

<p align="center">
  <a href="#-快速開始">快速開始</a> •
  <a href="#-課程目錄">課程目錄</a> •
  <a href="#-學習路線">學習路線</a> •
  <a href="#-實戰案例">實戰案例</a> •
  <a href="#-常見問題">FAQ</a>
</p>

---

## 📖 課程簡介

本課程是一套**完整的 PyTorch 中文教學**，涵蓋從最基礎的 Tensor 運算到模型部署的所有核心知識。每一章都包含**白話文原理說明**、**ASCII 架構圖解**、**逐行註解的程式碼**以及**可直接執行的實戰案例**。

### 為什麼選擇這個課程？

| 特色 | 說明 |
|------|------|
| 🇹🇼 **全中文註解** | 所有程式碼都有中文逐行說明，降低英文門檻 |
| 🎯 **由淺入深** | 從 `Hello World` 等級開始，到 GAN、遷移學習等進階主題 |
| 🔨 **動手實作** | 每章都是可直接 `python` 執行的完整程式，不只是片段 |
| 📊 **大量圖解** | ASCII 圖表 + 架構示意，用視覺理解抽象概念 |
| 🏭 **實戰導向** | 包含 6 個真實應用案例（影像、文字、生成、部署） |
| 🐛 **錯誤排查** | 專門整理初學者最常踩的 10 大陷阱與解法 |

### 你將學會

```
✅ 理解 Tensor 運算與自動微分原理
✅ 用 nn.Module 設計各種神經網路架構
✅ 掌握完整的模型訓練流程（資料→訓練→評估→部署）
✅ 實作 CNN 影像辨識（CIFAR-10 準確率 > 85%）
✅ 實作 LSTM 文字情感分析
✅ 使用預訓練模型做遷移學習
✅ 用 GAN 生成手寫數字圖片
✅ 將模型匯出為 ONNX 並部署成 REST API
```

---

## 🎯 適合對象

- 🐍 有基礎 Python 能力的開發者（了解 list、dict、class 即可）
- 🔄 想從 TensorFlow / Keras 轉換到 PyTorch 的使用者
- 📚 想理解深度學習原理（不只是呼叫 API）的初學者
- 🚀 想將 AI 模型部署到生產環境的工程師

### 先備知識

| 必備 | 建議 |
|------|------|
| Python 基礎語法 | NumPy 基本操作 |
| 了解 function / class | 線性代數基礎概念 |
| | 微積分（導數）概念 |

> 💡 **不需要深度學習經驗！** 課程會從頭解釋每一個概念。

---

## ⚡ 快速開始

### 1. 環境需求

```
Python   >= 3.9
PyTorch  >= 2.0
torchvision >= 0.15
```

### 2. 安裝步驟

```bash
# 方式一：pip 安裝（推薦）
pip install torch torchvision torchaudio matplotlib numpy

# 方式二：conda 安裝
conda install pytorch torchvision torchaudio -c pytorch

# 方式三：指定 CUDA 版本（有 NVIDIA GPU 時）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 取得課程

```bash
git clone https://github.com/ChunPingWang/pytorach-tutorial.git
cd pytorach-tutorial
```

### 4. 開始學習

```bash
# 從第一章開始
python chapters/01_tensors.py

# 或跳到任何一章（每章獨立可執行）
python chapters/05_cnn_image_classification.py
```

### 5. 驗證安裝

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
# 沒有 GPU 也可以完成所有課程，只是訓練速度較慢
```

---

## 📚 課程目錄

### 第一部分：基礎概念

| 章節 | 主題 | 核心內容 | 行數 |
|:----:|------|----------|:----:|
| [01](chapters/01_tensors.py) | **Tensor 張量基礎** | 建立 Tensor、基本運算、索引切片、形狀操作、Broadcasting、GPU 加速、NumPy 互轉 | 464 |
| [02](chapters/02_autograd.py) | **自動微分 Autograd** | requires_grad、計算圖、反向傳播、梯度累加陷阱、停止追蹤、用 Autograd 實作線性迴歸 | 342 |

### 第二部分：模型建構

| 章節 | 主題 | 核心內容 | 行數 |
|:----:|------|----------|:----:|
| [03](chapters/03_neural_networks.py) | **神經網路建構** | nn.Module、nn.Sequential、激活函數、CNN 基礎、Dropout/BatchNorm/Embedding、殘差連接 | 466 |
| [04](chapters/04_training_workflow.py) | **訓練工作流程** | Loss Function、Optimizer、Dataset/DataLoader、完整訓練迴圈、學習率排程、Early Stopping | 600 |

### 第三部分：實戰應用

| 章節 | 主題 | 核心內容 | 行數 |
|:----:|------|----------|:----:|
| [05](chapters/05_cnn_image_classification.py) | **實戰：CNN 影像辨識** | CIFAR-10 資料集、資料增強、CNN 架構設計、VGG 風格網路、類別準確率分析 | 434 |
| [06](chapters/06_nlp_text_classification.py) | **實戰：NLP 文字分類** | 文字前處理、詞嵌入、RNN/LSTM、雙向 LSTM 情感分析、Packed Sequences | 525 |
| [07](chapters/07_transfer_learning.py) | **遷移學習** | 預訓練模型載入、Feature Extraction、Fine-tuning、差異化學習率、ImageFolder | 431 |
| [08](chapters/08_gan.py) | **實戰：生成對抗網路** | GAN 原理、Generator/Discriminator 設計、對抗訓練、CGAN 條件式生成 | 448 |

### 第四部分：進階部署

| 章節 | 主題 | 核心內容 | 行數 |
|:----:|------|----------|:----:|
| [09](chapters/09_deployment.py) | **模型部署** | 模型儲存/載入/Checkpoint、TorchScript、ONNX 匯出、推論優化、Flask/FastAPI 服務化 | 536 |
| [10](chapters/10_best_practices.py) | **最佳實踐** | 10 大常見錯誤、記憶體管理、可重現性、訓練技巧集錦、專案結構、效能分析 | 617 |

> 📊 **總計 4,863 行**教學程式碼，每行都有中文註解。

---

## 🗺 學習路線

```
                          PyTorch 學習路線圖
 ═══════════════════════════════════════════════════════════

 第一階段：打好基礎（1-2 天）
 ┌─────────────┐     ┌──────────────┐
 │  Ch.1        │     │  Ch.2         │
 │  Tensor 運算 │ ──→ │  自動微分     │
 │  張量基礎    │     │  Autograd     │
 └─────────────┘     └──────────────┘
        │                    │
        └────────┬───────────┘
                 ▼
 第二階段：建構模型（2-3 天）
 ┌─────────────┐     ┌──────────────┐
 │  Ch.3        │     │  Ch.4         │
 │  nn.Module   │ ──→ │  訓練流程     │
 │  神經網路    │     │  Loss/Optim   │
 └─────────────┘     └──────────────┘
        │                    │
        └────────┬───────────┘
                 ▼
 第三階段：實戰應用（3-5 天）
 ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
 │  Ch.5     │ │  Ch.6     │ │  Ch.7     │ │  Ch.8     │
 │  CNN      │ │  LSTM     │ │  遷移學習 │ │  GAN      │
 │  影像辨識 │ │  文字分類 │ │  預訓練   │ │  生成模型 │
 └──────────┘ └──────────┘ └──────────┘ └──────────┘
        │            │            │            │
        └────────────┴─────┬──────┴────────────┘
                           ▼
 第四階段：部署上線（1-2 天）
 ┌─────────────┐     ┌──────────────┐
 │  Ch.9        │     │  Ch.10        │
 │  模型部署    │ ──→ │  最佳實踐     │
 │  ONNX/API   │     │  除錯技巧     │
 └─────────────┘     └──────────────┘

 ═══════════════════════════════════════════════════════════
 建議總學習時間：7-12 天（每天 2-3 小時）
```

### 依照目標選擇起點

| 你的目標 | 建議路線 |
|----------|----------|
| 完全零基礎，想學深度學習 | Ch.1 → 2 → 3 → 4 → 5（按順序） |
| 會 NumPy，想快速上手 PyTorch | Ch.1（快速瀏覽）→ 3 → 4 → 5 |
| 只想做影像辨識 | Ch.1 → 3 → 4 → 5 → 7 |
| 只想做 NLP 文字處理 | Ch.1 → 3 → 4 → 6 |
| 想了解 GAN 生成模型 | Ch.1 → 3 → 4 → 8 |
| 想部署現有模型 | Ch.9 → 10 |
| 從 TensorFlow 轉過來 | Ch.1（對比差異）→ 3 → 4 → 挑一個實戰 |

---

## 🏗 實戰案例總覽

本課程包含 **6 個完整的實戰案例**，每個都可以直接執行：

### 案例 1：手動實作線性迴歸（Ch.1 + Ch.2）

```
📌 任務：用房屋面積預測房價
📌 原理：純 Tensor 運算 → Autograd 自動計算梯度
📌 學到：梯度下降的完整數學原理
```

```python
# 模型：price = weight × area + bias
# 目標：學習到 weight ≈ 5.0, bias ≈ 200.0
predictions = weight * areas + bias
loss = ((predictions - prices) ** 2).mean()
loss.backward()  # 自動計算梯度！
```

### 案例 2：Iris 鳶尾花分類（Ch.4）

```
📌 任務：用 4 個特徵（花萼/花瓣長寬）分類 3 種花
📌 模型：3 層全連接神經網路
📌 學到：完整訓練流程 — DataLoader → Train → Evaluate
📌 成果：測試準確率 > 90%
```

### 案例 3：CIFAR-10 影像辨識（Ch.5）

```
📌 任務：辨識 10 類物體（飛機、汽車、鳥、貓…）
📌 模型：自訂 CNN + BatchNorm + Dropout
📌 資料：50,000 張 32×32 彩色圖片
📌 技巧：資料增強（翻轉、裁切、色彩抖動）
📌 進階：VGG 風格深度 CNN
```

```
輸入圖片 (3,32,32)
  → Conv(32) → BN → ReLU → Pool     # 偵測邊緣紋理
  → Conv(64) → BN → ReLU → Pool     # 偵測形狀圖案
  → Conv(128) → BN → ReLU → Pool    # 偵測物體部件
  → Flatten → FC(256) → Dropout
  → FC(10) → 預測類別
```

### 案例 4：LSTM 情感分析（Ch.6）

```
📌 任務：判斷電影評論是正面還是負面
📌 模型：Embedding → 雙向 LSTM → FC 分類器
📌 流程：文字 → 分詞 → 編碼 → Embedding → LSTM → 預測
📌 進階：Packed Sequences 處理變長序列
```

```
"This movie is wonderful" → [42, 7, 3, 88] → 🔢 → LSTM → 正面 (0.95)
"Terrible waste of time"  → [15, 62, 9, 23] → 🔢 → LSTM → 負面 (0.12)
```

### 案例 5：GAN 生成手寫數字（Ch.8）

```
📌 任務：從隨機雜訊生成逼真的手寫數字圖片
📌 模型：Generator（造假者）vs Discriminator（鑑定師）
📌 原理：兩個網路互相對抗，共同進步
📌 進階：條件式 GAN — 指定要生成哪個數字
```

```
隨機雜訊 z (100維)
  → Generator → 假圖片 (1,28,28) → Discriminator → 真/假？
                                         ↑
                  真實 MNIST 圖片 ────────┘
```

### 案例 6：遷移學習花卉分類（Ch.7）

```
📌 任務：用少量圖片分類 5 種花卉
📌 模型：預訓練 ResNet18 + 自訂分類頭
📌 策略：Feature Extraction / Fine-tuning
📌 技巧：差異化學習率 — 不同層用不同學習率
```

---

## 📂 目錄結構

```
pytorach-tutorial/
│
├── 📄 README.md                              ← 你正在看的這個檔案
│
└── 📁 chapters/                              ← 所有教學章節
    ├── 01_tensors.py                         # Tensor 張量基礎
    │   ├── 建立 Tensor（list / NumPy / 特殊矩陣）
    │   ├── 資料型別（dtype）與型別轉換
    │   ├── 基本運算（逐元素 / 統計 / 矩陣乘法）
    │   ├── 索引、切片、布林遮罩
    │   ├── 形狀操作（reshape / view / squeeze / permute / cat）
    │   ├── Broadcasting 廣播機制
    │   ├── GPU 加速
    │   └── 實際案例：手動線性迴歸
    │
    ├── 02_autograd.py                        # 自動微分 Autograd
    │   ├── requires_grad 與計算圖
    │   ├── 反向傳播 backward()
    │   ├── 梯度累加陷阱 ⚠️
    │   ├── 停止追蹤（no_grad / detach / inference_mode）
    │   └── 實際案例：Autograd 線性迴歸
    │
    ├── 03_neural_networks.py                 # nn.Module 神經網路
    │   ├── nn.Linear 全連接層
    │   ├── 激活函數（ReLU / Sigmoid / Softmax）
    │   ├── nn.Module vs nn.Sequential
    │   ├── 常用層（Dropout / BatchNorm / Embedding）
    │   ├── CNN 基礎（Conv2d / MaxPool2d）
    │   └── 進階：殘差連接 ResidualBlock
    │
    ├── 04_training_workflow.py               # 訓練工作流程
    │   ├── 損失函數（MSE / CrossEntropy / BCE）
    │   ├── 優化器（SGD / Adam / AdamW）
    │   ├── Dataset + DataLoader
    │   ├── 完整訓練迴圈（5 步驟）
    │   ├── 學習率排程（Step / Cosine / Plateau）
    │   ├── 完整框架（驗證 + 早停 + 排程）
    │   └── 實際案例：Iris 鳶尾花分類
    │
    ├── 05_cnn_image_classification.py        # 實戰：CNN 影像辨識
    │   ├── torchvision 資料載入
    │   ├── 資料增強（transforms）
    │   ├── CIFAR10CNN 模型設計
    │   ├── 完整訓練 + 測試
    │   ├── 每類別準確率分析
    │   └── 進階：VGG 風格深度 CNN
    │
    ├── 06_nlp_text_classification.py         # 實戰：NLP 文字分類
    │   ├── 文字前處理（分詞 / 詞彙表 / 編碼）
    │   ├── Word Embedding 詞嵌入
    │   ├── RNN / LSTM 原理
    │   ├── 雙向 LSTM 情感分析器
    │   └── 進階：Packed Sequences
    │
    ├── 07_transfer_learning.py               # 遷移學習
    │   ├── 載入預訓練 ResNet18
    │   ├── Feature Extraction（凍結策略）
    │   ├── Fine-tuning（微調策略）
    │   ├── 差異化學習率
    │   ├── ImageFolder 載入自己的資料
    │   └── 實戰：花卉分類
    │
    ├── 08_gan.py                             # 實戰：GAN
    │   ├── Generator 生成器（ConvTranspose2d）
    │   ├── Discriminator 判別器
    │   ├── 對抗訓練流程
    │   ├── 訓練技巧（標籤平滑 / 學習率）
    │   └── 進階：Conditional GAN
    │
    ├── 09_deployment.py                      # 模型部署
    │   ├── 模型儲存（state_dict / checkpoint）
    │   ├── TorchScript（trace / script）
    │   ├── ONNX 匯出 + Runtime 推論
    │   ├── 推論優化（compile / FP16 / batch）
    │   ├── Flask API 範例
    │   └── FastAPI API 範例
    │
    └── 10_best_practices.py                  # 最佳實踐
        ├── 10 大常見錯誤 ⚠️
        ├── 記憶體管理（梯度累積 / 混合精度）
        ├── 可重現性（seed 設定）
        ├── 訓練技巧集錦
        ├── 專案結構範例
        └── 延伸學習資源
```

---

## 🧠 核心概念速查表

### Tensor 運算

```python
# 建立
x = torch.tensor([1, 2, 3])
x = torch.zeros(3, 4)
x = torch.rand(3, 4)

# 形狀操作
x.reshape(2, -1)          # 改變形狀
x.unsqueeze(0)             # 增加維度
x.squeeze()                # 移除大小為 1 的維度
x.permute(2, 0, 1)         # 重排維度
x.flatten(1)               # 攤平

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
```

### 模型建構

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
```

### 訓練流程

```python
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()          # 1. 清除梯度
        outputs = model(inputs)        # 2. 前向傳播
        loss = criterion(outputs, labels)  # 3. 計算損失
        loss.backward()                # 4. 反向傳播
        optimizer.step()               # 5. 更新參數

    model.eval()
    with torch.no_grad():
        # 驗證/測試...
```

### 模型部署

```python
# 儲存
torch.save(model.state_dict(), 'model.pth')

# 載入
model.load_state_dict(torch.load('model.pth', weights_only=True))

# ONNX 匯出
torch.onnx.export(model, dummy_input, 'model.onnx')

# TorchScript
traced = torch.jit.trace(model, dummy_input)
traced.save('model.pt')
```

---

## ⚠️ 常見錯誤速查

| # | 錯誤 | 症狀 | 解法 |
|---|------|------|------|
| 1 | 忘記 `optimizer.zero_grad()` | Loss 不下降或震盪 | 每次 backward 前清零 |
| 2 | 忘記 `model.eval()` | 測試結果不穩定 | 測試前設定 eval 模式 |
| 3 | 裝置不匹配 | `Expected all tensors on same device` | 模型和資料都 `.to(device)` |
| 4 | CrossEntropy 標籤型別錯 | `expected Long` | 標籤用 `dtype=torch.long` |
| 5 | 多加了 Softmax | 準確率反而下降 | CrossEntropyLoss 已含 Softmax |
| 6 | in-place 操作 | `leaf Variable modified` | 用 `x = x + 1` 而非 `x += 1` |
| 7 | 轉 NumPy 忘記 detach | `Can't call numpy()` | `.detach().cpu().numpy()` |
| 8 | Windows num_workers | 程式直接卡住 | 設 `num_workers=0` |
| 9 | 學習率太大 | Loss 變成 NaN | 降低學習率或用 Adam |
| 10 | 形狀不匹配 | `size mismatch` | 在 forward() 印出每層 shape |

> 📖 完整說明與解法請見 [第十章](chapters/10_best_practices.py)

---

## ❓ 常見問題

<details>
<summary><b>Q: 沒有 GPU 可以學嗎？</b></summary>

可以！所有章節都能在 CPU 上執行。GPU 只是讓訓練速度更快，不影響學習。建議可以用 [Google Colab](https://colab.research.google.com/)（免費 GPU）來練習較大的模型。
</details>

<details>
<summary><b>Q: 需要多少數學基礎？</b></summary>

知道以下概念就足夠了：
- **矩陣乘法**：兩個表格相乘
- **導數/微分**：函數的變化率（梯度）
- **機率**：softmax 輸出的意義

課程中的數學都會用白話文解釋，不需要推導公式。
</details>

<details>
<summary><b>Q: PyTorch vs TensorFlow 該選哪個？</b></summary>

- **PyTorch**：更 Pythonic、Debug 容易、學術界主流、程式碼直覺
- **TensorFlow**：生產部署工具多、TFLite 行動端支援好

目前 PyTorch 已成為學術界和多數企業的首選。
</details>

<details>
<summary><b>Q: 每章大概要學多久？</b></summary>

| 章節 | 預估時間 |
|------|----------|
| Ch.1-2（基礎） | 各 1-2 小時 |
| Ch.3-4（建構） | 各 2-3 小時 |
| Ch.5-8（實戰） | 各 3-4 小時 |
| Ch.9-10（部署） | 各 1-2 小時 |

**總計約 20-30 小時**，建議每天 2-3 小時，約 7-12 天完成。
</details>

<details>
<summary><b>Q: 程式執行時下載資料集失敗怎麼辦？</b></summary>

Ch.5 和 Ch.8 會自動下載 CIFAR-10 / MNIST 資料集。如果網路問題導致下載失敗，程式會自動切換成模擬資料繼續教學，不影響學習。
</details>

---

## 📖 延伸學習資源

### 官方資源

- [PyTorch 官方文件](https://pytorch.org/docs/stable/)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [PyTorch 論壇](https://discuss.pytorch.org/)

### 推薦書籍

- 《Deep Learning with PyTorch》— Eli Stevens et al.
- 《Dive into Deep Learning》— [d2l.ai](https://d2l.ai/)（免費線上書）

### 進階主題

完成本課程後，建議的下一步學習方向：

```
本課程 (基礎 + 實戰)
  │
  ├─→ Transformer 架構 (NLP / Vision)
  │     └─→ HuggingFace Transformers
  │
  ├─→ 生成式 AI
  │     ├─→ Diffusion Models (Stable Diffusion)
  │     └─→ VAE (Variational Autoencoder)
  │
  ├─→ 分散式訓練
  │     ├─→ DistributedDataParallel
  │     └─→ DeepSpeed / FSDP
  │
  └─→ 模型壓縮
        ├─→ 量化 (Quantization)
        ├─→ 剪枝 (Pruning)
        └─→ 知識蒸餾 (Distillation)
```

### 實戰平台

- [Kaggle](https://www.kaggle.com/) — 比賽 + 免費 GPU
- [Google Colab](https://colab.research.google.com/) — 免費 GPU Notebook
- [Hugging Face](https://huggingface.co/) — 預訓練模型庫

---

## 🤝 貢獻

歡迎提交 Issue 或 Pull Request！

- 🐛 發現錯誤 → 開 Issue
- 💡 新增內容 → 提交 PR
- ⭐ 覺得有幫助 → 給個 Star

---

## 📜 授權

本專案採用 MIT License 授權。

---

<p align="center">
  <b>如果這個教學對你有幫助，請給個 ⭐ Star 支持！</b>
</p>

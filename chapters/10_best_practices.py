"""
=============================================================================
第十章：最佳實踐與常見錯誤
=============================================================================

經過前九章的學習，你已經掌握了 PyTorch 的核心技能。
本章整理了最重要的「經驗法則」和「常見陷阱」，
幫助你在實際專案中少走彎路。

本章內容：
─────────
✓ 10 個最常見的 PyTorch 錯誤
✓ 模型除錯技巧
✓ 記憶體管理
✓ 可重現性（Reproducibility）
✓ 訓練技巧集錦
✓ 程式碼風格最佳實踐
✓ 延伸學習資源
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

print("=" * 60)
print("第十章：最佳實踐與常見錯誤")
print("=" * 60)



def get_device():
    """自動選擇運算裝置：NVIDIA CUDA → Apple Silicon MPS → CPU

    同一份程式碼在 Colab（CUDA GPU）、Mac（MPS GPU）、純 CPU 環境都能直接執行。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")      # NVIDIA GPU（Colab / Windows / Linux）
    if torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon GPU（M 系列 Mac）
    return torch.device("cpu")           # 都沒有就用 CPU，一樣跑得動


device = get_device()
print(f"使用裝置: {device}")

# ─────────────────────────────────────────────────────────────
# 10.1 十大最常見的 PyTorch 錯誤
# ─────────────────────────────────────────────────────────────
print("\n📌 10.1 十大最常見的 PyTorch 錯誤")
print("-" * 40)

# ── 錯誤 #1：忘記清零梯度 ──
print("\n❌ 錯誤 #1：忘記 optimizer.zero_grad()")
print("─" * 30)
model = nn.Linear(3, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
x = torch.rand(4, 3)
target = torch.rand(4, 1)

# 錯誤寫法（梯度會累加）
for i in range(3):
    output = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()
    # 忘記 optimizer.zero_grad()！
    print(f"  step {i}: grad = {model.weight.grad.abs().sum().item():.4f} (越來越大!)")

# 正確寫法
print("\n✓ 正確寫法：每次 backward 前清零")
for i in range(3):
    optimizer.zero_grad()           # ← 加這行！
    output = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    print(f"  step {i}: grad = {model.weight.grad.abs().sum().item():.4f} (正常)")


# ── 錯誤 #2：忘記 model.eval() ──
print("\n\n❌ 錯誤 #2：測試時忘記 model.eval()")
print("─" * 30)
print("""
  影響：Dropout 和 BatchNorm 在 train/eval 模式下行為不同
  - Dropout: train=隨機丟棄, eval=全部保留
  - BatchNorm: train=用batch統計, eval=用全域統計

  正確的測試流程：
  ```python
  model.eval()                    # ← 切換到評估模式
  with torch.no_grad():           # ← 不計算梯度
      predictions = model(test_data)
  model.train()                   # ← 測試完切回訓練模式
  ```
""")

# 示範 Dropout 在不同模式下的差異
dropout_model = nn.Sequential(nn.Linear(10, 10), nn.Dropout(0.5), nn.Linear(10, 1))

x = torch.ones(1, 10)
dropout_model.train()
outputs_train = [dropout_model(x).item() for _ in range(5)]
print(f"  train 模式（每次不同）: {[f'{v:.3f}' for v in outputs_train]}")

dropout_model.eval()
outputs_eval = [dropout_model(x).item() for _ in range(5)]
print(f"  eval 模式（每次相同）: {[f'{v:.3f}' for v in outputs_eval]}")


# ── 錯誤 #3：裝置不匹配 ──
print("\n\n❌ 錯誤 #3：模型和資料不在同一裝置")
print("─" * 30)
print("""
  常見錯誤訊息：
  RuntimeError: Expected all tensors to be on the same device

  正確做法：
  ```python
  device = get_device()          # cuda / mps / cpu 自動選
  model = model.to(device)
  for inputs, labels in dataloader:
      inputs = inputs.to(device)    # ← 資料也要搬到同一裝置
      labels = labels.to(device)
      outputs = model(inputs)
  ```
""")


# ── 錯誤 #4：CrossEntropyLoss 的標籤型別 ──
print("\n❌ 錯誤 #4：CrossEntropyLoss 標籤型別錯誤")
print("─" * 30)
criterion = nn.CrossEntropyLoss()
logits = torch.rand(4, 3)  # 4 筆資料，3 個類別

# 錯誤：標籤是 float
try:
    labels_wrong = torch.tensor([0.0, 1.0, 2.0, 1.0])  # float!
    loss = criterion(logits, labels_wrong)
except Exception as e:
    print(f"  float 標籤: {type(e).__name__}")

# 正確：標籤必須是 long (int64)
labels_correct = torch.tensor([0, 1, 2, 1], dtype=torch.long)
loss = criterion(logits, labels_correct)
print(f"  long 標籤: loss = {loss.item():.4f} ✓")


# ── 錯誤 #5：模型輸出加了多餘的 Softmax ──
print("\n\n❌ 錯誤 #5：CrossEntropyLoss + 額外的 Softmax")
print("─" * 30)
print("""
  CrossEntropyLoss 內部已經包含了 LogSoftmax！

  ❌ 錯誤：
  output = model(x)
  output = F.softmax(output, dim=1)  # 多餘！
  loss = criterion(output, labels)    # 會重複做 softmax

  ✓ 正確：
  output = model(x)                   # 直接用 raw logits
  loss = criterion(output, labels)    # CrossEntropyLoss 自己會處理
""")


# ── 錯誤 #6：in-place 操作破壞計算圖 ──
print("\n❌ 錯誤 #6：in-place 操作")
print("─" * 30)
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 安全的操作
y = x + 1       # 建立新 Tensor ✓
print(f"  安全操作 x + 1: y = {y.tolist()}")

# 危險的 in-place 操作
# x += 1         # 就地修改，可能破壞計算圖 ✗
# x.add_(1)      # 底線結尾的都是 in-place 操作
print("  in-place 操作（如 x += 1, x.add_(1)）可能導致 RuntimeError")
print("  解法：用 x = x + 1 代替 x += 1")


# ── 錯誤 #7：忘記 detach ──
print("\n\n❌ 錯誤 #7：Tensor 轉 NumPy 忘記 detach")
print("─" * 30)
x = torch.tensor([1.0, 2.0], requires_grad=True)

try:
    np_array = x.numpy()
except Exception as e:
    print(f"  直接 .numpy(): {e}")

np_array = x.detach().cpu().numpy()  # 正確做法
print(f"  .detach().cpu().numpy(): {np_array} ✓")


# ── 錯誤 #8：DataLoader 的 num_workers ──
print("\n\n❌ 錯誤 #8：Windows 上 num_workers > 0 出錯")
print("─" * 30)
print("""
  Windows 上使用 num_workers > 0 可能會報錯。
  解決方案：
  1. 設 num_workers=0（最安全）
  2. 把 DataLoader 的程式碼放在 if __name__ == '__main__': 裡面
  3. 設定 persistent_workers=True
""")


# ── 錯誤 #9：學習率太大 ──
print("\n❌ 錯誤 #9：學習率太大導致 loss 爆炸")
print("─" * 30)
model = nn.Linear(10, 1)
x = torch.rand(32, 10)
y = torch.rand(32, 1)

for lr in [0.001, 0.1, 1.0, 10.0]:
    temp_model = nn.Linear(10, 1)
    optimizer = optim.SGD(temp_model.parameters(), lr=lr)

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        loss = ((temp_model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    trend = "✓ 收斂" if losses[-1] < losses[0] else "✗ 爆炸/不穩定"
    print(f"  lr={lr:5.3f}: loss {losses[0]:.3f} → {losses[-1]:.3f} {trend}")


# ── 錯誤 #10：形狀不匹配 ──
print("\n\n❌ 錯誤 #10：Tensor 形狀不匹配")
print("─" * 30)
print("""
  最常見的 Debug 技巧：在 forward() 裡印出每一層的形狀

  class MyModel(nn.Module):
      def forward(self, x):
          print(f"Input: {x.shape}")
          x = self.conv1(x)
          print(f"After conv1: {x.shape}")
          x = self.pool(x)
          print(f"After pool: {x.shape}")
          x = x.flatten(1)
          print(f"After flatten: {x.shape}")
          x = self.fc(x)
          print(f"After fc: {x.shape}")
          return x
""")


# ─────────────────────────────────────────────────────────────
# 10.2 記憶體管理
# ─────────────────────────────────────────────────────────────
print("\n\n📌 10.2 記憶體管理")
print("-" * 40)

print("""
  GPU 記憶體不夠（CUDA Out of Memory）的解決方案：

  1. 減小 batch_size（最簡單有效）

  2. 使用梯度累積（Gradient Accumulation）
     等同於用更大的 batch_size，但記憶體不會增加
""")

# 梯度累積範例
model = nn.Linear(100, 10)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
accumulation_steps = 4  # 累積 4 個 mini-batch 的梯度

print("梯度累積範例（等效 batch_size = 8 * 4 = 32）：")
optimizer.zero_grad()
for i in range(accumulation_steps):
    # 模擬一個 mini-batch
    x = torch.rand(8, 100)
    y = torch.randint(0, 10, (8,))

    output = model(x)
    loss = criterion(output, y) / accumulation_steps  # 除以累積步數
    loss.backward()        # 梯度會自動累加

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()   # 每累積 4 步才更新一次
        optimizer.zero_grad()
        print(f"  第 {i+1} 步：更新參數！")
    else:
        print(f"  第 {i+1} 步：累積梯度...")

print("""
  3. 混合精度訓練（Mixed Precision）
     用 FP16 做前向/反向傳播，FP32 做參數更新
     可以省一半的 GPU 記憶體
""")

# 混合精度範例
# 注意：GradScaler（loss 縮放）目前只有 CUDA 支援，
#      MPS 可以用 autocast，但不需要也不能用 GradScaler
print("混合精度訓練範例：")
if device.type in ('cuda', 'mps'):
    model = nn.Linear(100, 10).to(device)
    optimizer = optim.Adam(model.parameters())
    use_scaler = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type) if use_scaler else None

    x = torch.rand(32, 100).to(device)
    y = torch.randint(0, 10, (32,)).to(device)

    optimizer.zero_grad()
    with torch.amp.autocast(device.type):  # 自動選擇 FP16/FP32
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)

    if use_scaler:
        scaler.scale(loss).backward()      # 縮放 loss 防止 FP16 underflow
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()                    # MPS：直接 backward，不用縮放
        optimizer.step()
    print(f"  混合精度 loss（{device.type}）: {loss.item():.4f}")
else:
    print("  （需要 GPU — CUDA 或 MPS — 才能使用混合精度）")

print("""
  4. 其他記憶體節省技巧：
     - del tensor 之後清快取：
       CUDA → torch.cuda.empty_cache()   MPS → torch.mps.empty_cache()
     - 使用 checkpoint（犧牲速度換記憶體）
       from torch.utils.checkpoint import checkpoint
     - 減小模型（用較少的 channels/layers）
     - 減小圖片解析度
""")


# ─────────────────────────────────────────────────────────────
# 10.3 可重現性（Reproducibility）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 10.3 可重現性（Reproducibility）")
print("-" * 40)

def set_seed(seed=42):
    """設定所有隨機種子，確保結果可重現（CUDA / MPS / CPU 都適用）"""
    torch.manual_seed(seed)           # CPU，也會連帶設定各後端的預設種子
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)      # 多 GPU
        torch.backends.cudnn.deterministic = True   # cuDNN 專屬，MPS 沒有
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(42)
print("隨機種子已設定為 42")

# 驗證可重現性
set_seed(42)
a = torch.rand(3)
set_seed(42)
b = torch.rand(3)
print(f"  第一次: {a.tolist()}")
print(f"  第二次: {b.tolist()}")
print(f"  相同？{torch.equal(a, b)}")

print("""
  注意事項：
  - cudnn.deterministic=True 會降低 GPU 計算速度
  - 在開發/除錯時使用，正式訓練時可以關閉
  - DataLoader 的 worker_init_fn 也需要設定種子
  - 多 GPU 訓練的可重現性更難保證
""")


# ─────────────────────────────────────────────────────────────
# 10.4 訓練技巧集錦
# ─────────────────────────────────────────────────────────────
print("\n\n📌 10.4 訓練技巧集錦")
print("-" * 40)

print("""
  1. 資料正規化
     ────────────
     永遠在訓練前對資料做標準化
     方式一：x = (x - mean) / std
     方式二：x = x / 255.0  （圖片用）

  2. 權重初始化
     ────────────
     PyTorch 預設的初始化通常夠好（Kaiming/Xavier）
     如果要自訂：
""")

# 自訂權重初始化
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 10))
model.apply(init_weights)
print("  自訂初始化完成")

print("""
  3. Gradient Clipping（梯度裁切）
     ─────────────────────────────
     防止梯度爆炸（特別是 RNN/LSTM）
""")

model = nn.LSTM(10, 20, batch_first=True)
optimizer = optim.Adam(model.parameters())
x = torch.rand(4, 5, 10)
output, _ = model(x)
loss = output.sum()
loss.backward()

# 裁切梯度（norm 不超過 1.0）
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"  裁切前的梯度 norm: {total_norm:.4f}")

print("""
  4. Label Smoothing
     ────────────────
     把 hard label [0, 0, 1, 0] 變成 soft label [0.025, 0.025, 0.925, 0.025]
     減少過擬合，提高泛化能力
""")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
logits = torch.rand(4, 10)
labels = torch.randint(0, 10, (4,))
loss = criterion(logits, labels)
print(f"  Label Smoothing loss: {loss.item():.4f}")

print("""
  5. Early Stopping
     ──────────────
     當驗證 loss 不再改善時，停止訓練
     防止過擬合（詳見第四章的完整框架）

  6. 模型集成（Ensemble）
     ──────────────────
     訓練多個模型，取平均或投票
     通常能提升 1~3% 的準確率

  7. 常用超參數搜索範圍
     ──────────────────
     ┌──────────────┬────────────────────┐
     │ 學習率        │ 1e-4 ~ 1e-2       │
     │ batch_size   │ 16, 32, 64, 128    │
     │ weight_decay │ 1e-5 ~ 1e-2       │
     │ dropout      │ 0.1 ~ 0.5         │
     │ hidden_dim   │ 64, 128, 256, 512  │
     └──────────────┴────────────────────┘
""")


# ─────────────────────────────────────────────────────────────
# 10.5 程式碼組織最佳實踐
# ─────────────────────────────────────────────────────────────
print("\n\n📌 10.5 程式碼組織最佳實踐")
print("-" * 40)

print("""
  推薦的專案結構：
  ```
  project/
  ├── config/
  │   └── config.yaml          # 超參數設定
  ├── data/
  │   ├── dataset.py           # 資料集定義
  │   └── transforms.py        # 前處理
  ├── models/
  │   ├── __init__.py
  │   ├── resnet.py            # 模型定義
  │   └── losses.py            # 自訂損失函數
  ├── utils/
  │   ├── metrics.py           # 評估指標
  │   ├── visualization.py     # 視覺化
  │   └── seed.py              # 隨機種子
  ├── train.py                 # 訓練腳本
  ├── evaluate.py              # 評估腳本
  ├── predict.py               # 推論腳本
  └── requirements.txt         # 依賴套件
  ```

  設定檔範例 (config.yaml)：
  ```yaml
  model:
    name: resnet18
    num_classes: 10
    pretrained: true

  training:
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    weight_decay: 0.0001
    optimizer: adam
    scheduler: cosine

  data:
    train_dir: ./data/train
    val_dir: ./data/val
    image_size: 224
    augmentation: true
  ```
""")


# ─────────────────────────────────────────────────────────────
# 10.6 效能分析工具
# ─────────────────────────────────────────────────────────────
print("\n\n📌 10.6 效能分析工具")
print("-" * 40)

# 用 torch.profiler 分析效能瓶頸
print("PyTorch Profiler 範例：")
model = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 10))
x = torch.rand(64, 100)

try:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            _ = model(x)

    # 印出最耗時的操作
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
except Exception as e:
    print(f"  Profiler 不可用: {e}")

# 記憶體使用量追蹤（CUDA 和 MPS 的 API 不一樣）
if device.type == 'cuda':
    print(f"\nGPU 記憶體使用（CUDA）：")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"  快取:   {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
elif device.type == 'mps':
    print(f"\nGPU 記憶體使用（MPS）：")
    print(f"  已分配: {torch.mps.current_allocated_memory() / 1024**2:.1f} MB")
    print(f"  驅動配置: {torch.mps.driver_allocated_memory() / 1024**2:.1f} MB")


# ─────────────────────────────────────────────────────────────
# 10.7 延伸學習資源
# ─────────────────────────────────────────────────────────────
print("\n\n📌 10.7 延伸學習資源")
print("-" * 40)

print("""
  官方資源：
  ─────────
  • PyTorch 官方文件: pytorch.org/docs
  • PyTorch 官方教程: pytorch.org/tutorials
  • PyTorch 論壇: discuss.pytorch.org

  進階主題：
  ─────────
  1. Transformer 架構
     - Self-Attention 機制
     - Vision Transformer (ViT)
     - 使用 HuggingFace Transformers 套件

  2. 生成式模型
     - VAE (Variational Autoencoder)
     - Diffusion Models（Stable Diffusion 的核心）
     - Flow-based Models

  3. 強化學習
     - DQN, PPO, A3C
     - 使用 PyTorch + Gymnasium

  4. 分散式訓練
     - DataParallel（單機多 GPU）
     - DistributedDataParallel（多機多 GPU）
     - FSDP（Fully Sharded Data Parallel）
     - DeepSpeed / Megatron-LM

  5. 模型壓縮
     - 量化（Quantization）
     - 剪枝（Pruning）
     - 知識蒸餾（Knowledge Distillation）

  推薦書籍：
  ─────────
  • 《Deep Learning with PyTorch》— Eli Stevens et al.
  • 《Dive into Deep Learning》— 免費線上書（d2l.ai）

  實戰平台：
  ─────────
  • Kaggle（比賽 + 免費 GPU）
  • Google Colab（免費 GPU notebook）
  • Hugging Face（預訓練模型庫）
""")


# ─────────────────────────────────────────────────────────────
# 10.8 課程總結
# ─────────────────────────────────────────────────────────────
print("\n\n📌 10.8 課程總結")
print("-" * 40)

print("""
  恭喜！你已經完成了 PyTorch 完整教學課程！

  回顧所學：
  ═════════

  基礎概念（Ch.1-2）
  ├── Tensor 張量：PyTorch 的核心資料結構
  └── Autograd 自動微分：梯度計算的引擎

  模型建構（Ch.3-4）
  ├── nn.Module：模組化的神經網路
  └── 訓練流程：Loss → Backward → Optimize

  實戰應用（Ch.5-8）
  ├── CNN：影像辨識（CIFAR-10）
  ├── RNN/LSTM：文字分類（情感分析）
  ├── Transfer Learning：遷移學習（花卉分類）
  └── GAN：生成對抗網路（手寫數字生成）

  進階部署（Ch.9-10）
  ├── 模型儲存/載入/匯出
  ├── TorchScript / ONNX
  ├── API 服務化
  └── 最佳實踐與除錯技巧

  下一步行動建議：
  ─────────────────
  1. 選一個感興趣的 Kaggle 比賽，用所學知識參賽
  2. 嘗試用 PyTorch 復現一篇論文
  3. 深入學習 Transformer 架構
  4. 探索生成式 AI（Diffusion Models）
  5. 貢獻 PyTorch 開源社群

  祝你在深度學習的旅途上一切順利！
""")

print("=" * 60)
print("課程結束！感謝學習！")
print("=" * 60)

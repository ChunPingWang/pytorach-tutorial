"""
=============================================================================
第四章：訓練工作流程 — Loss, Optimizer, Training Loop
=============================================================================

深度學習訓練的完整流程：
──────────────────────

  ┌─────────────────────────────────────────────────────────┐
  │                    訓練迴圈 (Epoch)                      │
  │                                                         │
  │  1. optimizer.zero_grad()    ← 清除上一步的梯度           │
  │  2. outputs = model(inputs)  ← 前向傳播                  │
  │  3. loss = criterion(outputs, labels)  ← 計算損失        │
  │  4. loss.backward()          ← 反向傳播（計算梯度）        │
  │  5. optimizer.step()         ← 更新參數                  │
  │                                                         │
  │  重複以上步驟，直到 loss 收斂                              │
  └─────────────────────────────────────────────────────────┘

三大核心元素：
─────────────
1. Loss Function（損失函數）：衡量「預測值」和「真實值」的差距
2. Optimizer（優化器）：決定參數要怎麼更新（方向 + 步長）
3. DataLoader：有效率地把資料分批餵給模型

本章學習目標：
─────────────
✓ 常用的損失函數
✓ 常用的優化器
✓ Dataset 和 DataLoader 的使用
✓ 完整的訓練迴圈
✓ 驗證與評估
✓ 學習率排程
✓ 實際案例：完整訓練一個分類模型
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

print("=" * 60)
print("第四章：訓練工作流程")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 4.1 損失函數（Loss Function）
# ─────────────────────────────────────────────────────────────
print("\n📌 4.1 損失函數")
print("-" * 40)

# 損失函數 = 衡量模型預測有多「差」
# Loss 越小 → 預測越接近真實值 → 模型越好

# ── 回歸問題常用 ──

# MSELoss（均方誤差）：預測連續數值（價格、溫度）
# MSE = mean((predicted - actual)²)
mse_loss = nn.MSELoss()
predicted = torch.tensor([2.5, 3.0, 4.5])
actual = torch.tensor([3.0, 3.0, 5.0])
loss = mse_loss(predicted, actual)
print(f"MSELoss: {loss.item():.4f}")
# = ((2.5-3)² + (3-3)² + (4.5-5)²) / 3 = (0.25 + 0 + 0.25) / 3 = 0.1667

# L1Loss（平均絕對誤差）：對離群值較不敏感
# MAE = mean(|predicted - actual|)
l1_loss = nn.L1Loss()
loss = l1_loss(predicted, actual)
print(f"L1Loss: {loss.item():.4f}")

# ── 分類問題常用 ──

# CrossEntropyLoss：多類別分類（最常用！）
# 內部自動做 Softmax + NLLLoss
# 所以模型輸出不需要自己加 Softmax！
ce_loss = nn.CrossEntropyLoss()

# logits（模型原始輸出，未經 Softmax）
logits = torch.tensor([[2.0, 1.0, 0.1],    # 第 1 筆：模型覺得是 class 0
                        [0.5, 2.5, 0.3]])    # 第 2 筆：模型覺得是 class 1
labels = torch.tensor([0, 1])               # 真實標籤（必須是 long 型別）
loss = ce_loss(logits, labels)
print(f"\nCrossEntropyLoss: {loss.item():.4f}")

# BCEWithLogitsLoss：二元分類
# 內部自動做 Sigmoid，所以輸出也不需要自己加
bce_loss = nn.BCEWithLogitsLoss()
logits = torch.tensor([0.8, -0.5, 1.2])    # 模型原始輸出
targets = torch.tensor([1.0, 0.0, 1.0])    # 真實標籤（0 或 1）
loss = bce_loss(logits, targets)
print(f"BCEWithLogitsLoss: {loss.item():.4f}")

print("""
  損失函數選擇指南：
  ┌──────────────┬───────────────────────┐
  │ 回歸問題      │ MSELoss (L2) / L1Loss │
  │ 多類別分類    │ CrossEntropyLoss      │
  │ 二元分類      │ BCEWithLogitsLoss     │
  │ 多標籤分類    │ BCEWithLogitsLoss     │
  └──────────────┴───────────────────────┘
""")


# ─────────────────────────────────────────────────────────────
# 4.2 優化器（Optimizer）
# ─────────────────────────────────────────────────────────────
print("\n📌 4.2 優化器")
print("-" * 40)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

# SGD：隨機梯度下降（最基本）
# 參數 = 參數 - learning_rate × 梯度
sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam：自適應學習率（最常用！）
# 結合了 Momentum 和 RMSprop 的優點
# 每個參數有自己的學習率，自動調整
adam = optim.Adam(model.parameters(), lr=0.001)

# AdamW：Adam + Weight Decay（目前推薦）
# Weight Decay 是一種正則化方法，防止參數變太大
adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

print(f"SGD 優化器: {sgd}")
print(f"Adam 優化器: {adam}")

print("""
  優化器選擇指南：
  ┌──────────┬──────────────────────────────────┐
  │ Adam     │ 首選！大部分情況表現良好            │
  │ AdamW    │ Adam 改良版，有 weight decay        │
  │ SGD      │ 配合 momentum，CV 任務常用          │
  │ RMSprop  │ RNN 任務有時表現較好                │
  └──────────┴──────────────────────────────────┘

  學習率參考：
  - Adam/AdamW: 1e-3 ~ 3e-4（起始值）
  - SGD: 0.01 ~ 0.1（起始值）
""")


# ─────────────────────────────────────────────────────────────
# 4.3 Dataset 和 DataLoader
# ─────────────────────────────────────────────────────────────
print("\n📌 4.3 Dataset 和 DataLoader")
print("-" * 40)

# 為什麼需要 DataLoader？
# - 把資料分成 batch（不能一次餵所有資料，記憶體會不夠）
# - 自動 shuffle（打亂順序，避免模型記住順序）
# - 多線程載入（加速資料讀取）

# 方法一：自訂 Dataset
class HousePriceDataset(Dataset):
    """
    自訂資料集：房價預測

    每筆資料包含：
    - features: [面積, 房間數, 屋齡, 距離捷運站, 樓層]
    - label: 房價（萬）
    """
    def __init__(self, num_samples=1000):
        torch.manual_seed(42)

        # 模擬 5 個特徵的資料
        self.features = torch.randn(num_samples, 5)

        # 模擬房價 = 各特徵的加權總和 + 雜訊
        true_weights = torch.tensor([30.0, 10.0, -5.0, -8.0, 3.0])
        self.labels = self.features @ true_weights + 500 + torch.randn(num_samples) * 20

    def __len__(self):
        """回傳資料集的大小（DataLoader 需要）"""
        return len(self.labels)

    def __getitem__(self, idx):
        """回傳第 idx 筆資料（DataLoader 需要）"""
        return self.features[idx], self.labels[idx]


# 建立資料集
dataset = HousePriceDataset(num_samples=1000)
print(f"資料集大小: {len(dataset)}")
print(f"第一筆資料: features={dataset[0][0].tolist()}, label={dataset[0][1].item():.2f}")

# 分割訓練集和測試集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print(f"\n訓練集: {len(train_dataset)} 筆")
print(f"測試集: {len(test_dataset)} 筆")

# 建立 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,       # 每批 32 筆
    shuffle=True,        # 每個 epoch 打亂順序
    num_workers=0,       # 子線程數（Windows 設 0 較安全）
    drop_last=True       # 丟棄最後不足一批的資料
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False         # 測試集不需要打亂
)

# 看看 DataLoader 怎麼迭代
for batch_idx, (features, labels) in enumerate(train_loader):
    print(f"\nBatch {batch_idx}: features 形狀={features.shape}, labels 形狀={labels.shape}")
    if batch_idx >= 2:   # 只看前 3 個 batch
        print("  ...")
        break

print(f"\n每個 epoch 有 {len(train_loader)} 個 batch")


# ─────────────────────────────────────────────────────────────
# 4.4 完整訓練迴圈
# ─────────────────────────────────────────────────────────────
print("\n\n📌 4.4 完整訓練迴圈（房價預測）")
print("-" * 40)

# ── 步驟一：定義模型 ──
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)     # 輸出一個數值（房價）
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # 移除最後一個維度

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HousePriceModel().to(device)

# ── 步驟二：定義損失函數和優化器 ──
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ── 步驟三：訓練迴圈 ──
num_epochs = 30
train_losses = []

print("開始訓練：")
for epoch in range(num_epochs):
    model.train()                       # 切換到訓練模式
    epoch_loss = 0.0
    num_batches = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        # 1. 清除梯度
        optimizer.zero_grad()

        # 2. 前向傳播
        predictions = model(features)

        # 3. 計算損失
        loss = criterion(predictions, labels)

        # 4. 反向傳播
        loss.backward()

        # 5. 更新參數
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    if epoch % 5 == 0:
        print(f"  Epoch {epoch:3d}/{num_epochs} | Train Loss: {avg_loss:.4f}")

# ── 步驟四：測試評估 ──
print("\n評估模型：")
model.eval()                            # 切換到評估模式
test_loss = 0.0
num_batches = 0

with torch.no_grad():                   # 不需要計算梯度
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        predictions = model(features)
        loss = criterion(predictions, labels)
        test_loss += loss.item()
        num_batches += 1

avg_test_loss = test_loss / num_batches
print(f"  Test Loss: {avg_test_loss:.4f}")

# 看看幾個預測結果
with torch.no_grad():
    sample_features, sample_labels = next(iter(test_loader))
    sample_features = sample_features.to(device)
    sample_preds = model(sample_features).cpu()
    print("\n  預測值 vs 真實值（前 5 筆）：")
    for i in range(5):
        print(f"    預測: {sample_preds[i].item():.1f}, 真實: {sample_labels[i].item():.1f}")


# ─────────────────────────────────────────────────────────────
# 4.5 學習率排程（Learning Rate Scheduler）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 4.5 學習率排程")
print("-" * 40)

print("""
為什麼要調整學習率？
- 開始時用大學習率：快速接近最佳解
- 後期用小學習率：精細調整，避免在最佳解附近震盪

圖解：
  Loss
  │╲
  │  ╲         大學習率快速下降
  │    ╲_
  │      ╲___          小學習率精細調整
  │          ╲_____
  └──────────────────── Epoch
""")

# 重建模型和優化器
model = HousePriceModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# StepLR：每隔 step_size 個 epoch，學習率乘以 gamma
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
print("StepLR：每 10 個 epoch 學習率減半")
for epoch in range(31):
    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: lr = {scheduler.get_last_lr()[0]:.6f}")
    scheduler.step()

# CosineAnnealingLR：餘弦退火（平滑地降低學習率）
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
print("\nCosineAnnealingLR：")
for epoch in range(51):
    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: lr = {scheduler.get_last_lr()[0]:.6f}")
    scheduler.step()

# ReduceLROnPlateau：當指標不再改善時降低學習率（最實用！）
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # 監控指標是越小越好（loss）
    factor=0.5,          # 學習率乘以 0.5
    patience=5,          # 連續 5 個 epoch 沒改善就降低
    verbose=True
)
print("\nReduceLROnPlateau：loss 不下降時自動降低學習率")
print("  用法：scheduler.step(val_loss)")


# ─────────────────────────────────────────────────────────────
# 4.6 完整的訓練框架（含驗證 + 早停 + 學習率排程）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 4.6 完整訓練框架")
print("-" * 40)

def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, num_epochs=50, patience=10, device='cpu'):
    """
    完整的訓練函數，包含：
    - 訓練/驗證分離
    - 早停（Early Stopping）
    - 學習率排程
    - 最佳模型保存
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # ── 訓練階段 ──
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ── 驗證階段 ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # 記錄歷史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # 學習率排程
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # 早停檢查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # 保存最佳模型
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter >= patience:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | Train: {avg_train_loss:.4f} "
                  f"| Val: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        if patience_counter >= patience:
            print(f"\n  早停！驗證 loss 連續 {patience} 個 epoch 沒有改善")
            break

    # 載入最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history

# 使用完整框架訓練
model = HousePriceModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss()

# 準備驗證集（從訓練集再分出一部分）
full_dataset = HousePriceDataset(1000)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

print("使用完整訓練框架：")
history = train_model(
    model, train_loader, val_loader, criterion, optimizer,
    scheduler=scheduler, num_epochs=100, patience=15, device=device
)
print(f"  訓練完成，最終 val_loss: {min(history['val_loss']):.4f}")


# ─────────────────────────────────────────────────────────────
# 4.7 分類問題的完整範例：Iris 鳶尾花
# ─────────────────────────────────────────────────────────────
print("\n\n📌 4.7 實際案例：Iris 鳶尾花分類")
print("-" * 40)

# 手動建立 Iris 資料集（避免需要 sklearn 依賴）
torch.manual_seed(42)
np.random.seed(42)

# 模擬三個類別的花：每類 50 筆，4 個特徵
# 特徵：花萼長度、花萼寬度、花瓣長度、花瓣寬度
class0 = torch.randn(50, 4) * 0.3 + torch.tensor([5.0, 3.4, 1.5, 0.2])
class1 = torch.randn(50, 4) * 0.3 + torch.tensor([5.9, 2.8, 4.3, 1.3])
class2 = torch.randn(50, 4) * 0.3 + torch.tensor([6.6, 3.0, 5.6, 2.0])

X = torch.cat([class0, class1, class2], dim=0)     # (150, 4)
y = torch.cat([torch.zeros(50), torch.ones(50), torch.full((50,), 2)]).long()  # (150,)

# 打亂資料
indices = torch.randperm(150)
X, y = X[indices], y[indices]

# 分割資料集
X_train, X_test = X[:120], X[120:]
y_train, y_test = y[:120], y[120:]

# 建立 DataLoader
train_data = torch.utils.data.TensorDataset(X_train, y_train)
test_data = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

# 建立分類模型
class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)    # 3 個類別
        )

    def forward(self, x):
        return self.net(x)

model = IrisClassifier().to(device)
criterion = nn.CrossEntropyLoss()       # 多類別分類
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 訓練
print("訓練 Iris 分類器：")
for epoch in range(100):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        # 計算準確率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = outputs.max(1)  # 取最大值的索引作為預測類別
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = 100.0 * correct / total
        print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Test Accuracy: {accuracy:.1f}%")

# 最終測試
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

print(f"\n最終測試準確率: {100.0 * correct / total:.1f}%")


# ─────────────────────────────────────────────────────────────
# 4.8 練習題
# ─────────────────────────────────────────────────────────────
print("\n\n📌 4.8 練習題")
print("-" * 40)
print("""
練習 1：把 Iris 分類器加入 Dropout 和 BatchNorm，觀察準確率變化

練習 2：試試不同的優化器（SGD vs Adam vs AdamW），比較收斂速度

練習 3：實作一個二元分類問題：
        - 模擬兩個高斯分佈的 2D 資料
        - 用 BCEWithLogitsLoss 訓練
        - 畫出決策邊界

練習 4：在訓練迴圈中加入 gradient clipping：
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        觀察對訓練穩定性的影響
""")

print("\n" + "=" * 60)
print("第四章結束！下一章：實戰 — CNN 影像辨識")
print("=" * 60)

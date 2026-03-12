"""
=============================================================================
第五章：實戰 — CNN 影像辨識（CIFAR-10）
=============================================================================

什麼是 CNN（卷積神經網路）？
───────────────────────────
CNN 是專門處理「有空間結構」的資料（圖片、影片）的神經網路。

全連接層 vs CNN：
─────────────────
全連接層：把圖片攤平成 1D → 失去「空間位置」資訊
CNN：    用小視窗在圖片上滑動 → 保留「空間位置」資訊

CNN 的核心概念：
──────────────
1. 卷積層（Conv）：用小 kernel 偵測特徵（邊緣、紋理、形狀）
2. 池化層（Pool）：縮小尺寸，保留重要特徵
3. 全連接層（FC）：最後做分類

圖解 CNN 處理流程：
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────┐   ┌────────┐
│ 輸入圖片  │ → │ 卷積+ReLU │ → │ 池化     │ → │ 攤平  │ → │ 全連接  │ → 預測
│ 32×32×3  │   │ 偵測邊緣  │   │ 縮小尺寸  │   │      │   │ 分類   │
└──────────┘   └──────────┘   └──────────┘   └──────┘   └────────┘

CIFAR-10 資料集：
────────────────
- 60,000 張 32×32 彩色圖片
- 10 個類別：飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船、卡車
- 訓練集 50,000 張 + 測試集 10,000 張

本章學習目標：
─────────────
✓ 使用 torchvision 載入標準資料集
✓ 圖片前處理（transforms）
✓ 建構 CNN 模型
✓ 完整訓練 + 評估流程
✓ 資料增強（Data Augmentation）
✓ 混淆矩陣分析
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

print("=" * 60)
print("第五章：實戰 — CNN 影像辨識（CIFAR-10）")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

# ─────────────────────────────────────────────────────────────
# 5.1 資料載入與前處理
# ─────────────────────────────────────────────────────────────
print("\n📌 5.1 資料載入與前處理")
print("-" * 40)

# Transforms：定義圖片的前處理流程
# 就像食材進廚房前要「洗、切、調味」

# 訓練集：加入資料增強（讓模型見過更多變化，減少過擬合）
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),    # 50% 機率水平翻轉
    transforms.RandomCrop(32, padding=4),       # 隨機裁切（先填充再裁）
    transforms.ColorJitter(brightness=0.2,      # 隨機調整亮度
                           contrast=0.2),       # 隨機調整對比
    transforms.ToTensor(),                      # 轉成 Tensor (0~1)
    transforms.Normalize(                       # 標準化
        mean=[0.4914, 0.4822, 0.4465],         # CIFAR-10 的均值
        std=[0.2470, 0.2435, 0.2616]           # CIFAR-10 的標準差
    )
])

# 測試集：只做基本前處理（不加資料增強！）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

print("""
  資料增強（Data Augmentation）的作用：
  ┌─────────────┬──────────────────────────────┐
  │ 水平翻轉     │ 讓模型學會「左邊的貓也是貓」    │
  │ 隨機裁切     │ 讓模型學會「只看到一部分也能認」  │
  │ 亮度調整     │ 讓模型學會「暗一點也是同一類」    │
  └─────────────┴──────────────────────────────┘
  結果：不需要更多真實圖片，就能有效減少過擬合！
""")

# 下載並載入 CIFAR-10
# 第一次執行會自動下載（約 170MB）
try:
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    DATA_AVAILABLE = True
except Exception as e:
    print(f"  下載資料集失敗（可能網路問題）：{e}")
    print("  將使用模擬資料繼續教學")
    DATA_AVAILABLE = False

    # 模擬資料（用於沒網路的情況）
    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000):
            self.data = torch.randn(size, 3, 32, 32)
            self.targets = torch.randint(0, 10, (size,))
        def __len__(self):
            return len(self.targets)
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    train_dataset = FakeDataset(5000)
    test_dataset = FakeDataset(1000)

# 類別名稱
classes = ('飛機', '汽車', '鳥', '貓', '鹿',
           '狗', '青蛙', '馬', '船', '卡車')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"訓練集: {len(train_dataset)} 張圖片")
print(f"測試集: {len(test_dataset)} 張圖片")
print(f"類別: {classes}")

# 看一個 batch 的資料
images, labels = next(iter(train_loader))
print(f"\n一個 batch 的形狀: {images.shape}")   # (64, 3, 32, 32)
print(f"一個 batch 的標籤: {labels[:10].tolist()}")


# ─────────────────────────────────────────────────────────────
# 5.2 建構 CNN 模型
# ─────────────────────────────────────────────────────────────
print("\n\n📌 5.2 建構 CNN 模型")
print("-" * 40)

class CIFAR10CNN(nn.Module):
    """
    CIFAR-10 分類器

    架構：
    Input (3,32,32)
      → Conv1 (32,32,32) → BN → ReLU → Pool (32,16,16)
      → Conv2 (64,16,16) → BN → ReLU → Pool (64,8,8)
      → Conv3 (128,8,8)  → BN → ReLU → Pool (128,4,4)
      → Flatten (2048)
      → FC1 (256) → ReLU → Dropout
      → FC2 (10)

    每一層的作用：
    - Conv 層：提取特徵（淺層=邊緣/紋理，深層=形狀/物體）
    - BN 層：穩定訓練、加速收斂
    - ReLU：加入非線性
    - Pool 層：減少計算量、增加感受野
    - Dropout：防止過擬合
    """
    def __init__(self):
        super().__init__()

        # 卷積區塊 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),     # (3,32,32) → (32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                               # (32,32,32) → (32,16,16)
        )

        # 卷積區塊 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # (32,16,16) → (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                               # (64,16,16) → (64,8,8)
        )

        # 卷積區塊 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # (64,8,8) → (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                               # (128,8,8) → (128,4,4)
        )

        # 分類頭
        self.classifier = nn.Sequential(
            nn.Flatten(),                                     # (128,4,4) → (2048)
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)                               # 10 個類別
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


model = CIFAR10CNN().to(device)
print(f"模型結構：\n{model}")

# 計算參數量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n總參數量: {total_params:,}")

# 驗證形狀
dummy = torch.rand(1, 3, 32, 32).to(device)
out = model(dummy)
print(f"輸入: {dummy.shape} → 輸出: {out.shape}")


# ─────────────────────────────────────────────────────────────
# 5.3 訓練模型
# ─────────────────────────────────────────────────────────────
print("\n\n📌 5.3 訓練模型")
print("-" * 40)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 15  # 完整訓練通常需要 50+ epochs，這裡用 15 做示範

print("開始訓練 CIFAR-10 分類器：")
for epoch in range(num_epochs):
    # ── 訓練 ──
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100.0 * correct / total
    avg_train_loss = train_loss / len(train_loader)

    # ── 測試 ──
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct / total
    avg_test_loss = test_loss / len(test_loader)

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    print(f"  Epoch {epoch+1:2d}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | "
          f"Test Loss: {avg_test_loss:.4f} Acc: {test_acc:.1f}% | "
          f"LR: {current_lr:.6f}")


# ─────────────────────────────────────────────────────────────
# 5.4 模型分析：每個類別的準確率
# ─────────────────────────────────────────────────────────────
print("\n\n📌 5.4 每個類別的準確率")
print("-" * 40)

class_correct = [0] * 10
class_total = [0] * 10

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

print("每個類別的準確率：")
for i in range(10):
    if class_total[i] > 0:
        acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  {classes[i]:>4s}: {acc:5.1f}% ({class_correct[i]:4d}/{class_total[i]:4d})")


# ─────────────────────────────────────────────────────────────
# 5.5 模型預測展示
# ─────────────────────────────────────────────────────────────
print("\n\n📌 5.5 模型預測範例")
print("-" * 40)

model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    _, predicted = outputs.max(1)

print("前 10 張圖片的預測結果：")
print(f"{'真實':>6s} | {'預測':>6s} | {'信心度':>6s} | {'結果':>4s}")
print("-" * 35)
for i in range(10):
    true_label = classes[labels[i].item()]
    pred_label = classes[predicted[i].item()]
    confidence = probs[i][predicted[i]].item() * 100
    correct = "✓" if predicted[i] == labels[i] else "✗"
    print(f"{true_label:>6s} | {pred_label:>6s} | {confidence:5.1f}% | {correct}")


# ─────────────────────────────────────────────────────────────
# 5.6 進階：更深的 CNN（VGG 風格）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 5.6 進階：VGG 風格的深度 CNN")
print("-" * 40)

class VGGStyleNet(nn.Module):
    """
    VGG 風格的 CNN — 使用連續的小 kernel (3x3) 堆疊

    核心理念：
    兩個 3x3 卷積 = 一個 5x5 卷積的感受野
    三個 3x3 卷積 = 一個 7x7 卷積的感受野
    但參數更少，非線性更多！
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 64
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),    # 32→16
            nn.Dropout2d(0.25),

            # Block 2: 64 → 128
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),    # 16→8
            nn.Dropout2d(0.25),

            # Block 3: 128 → 256
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),    # 8→4
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

vgg_model = VGGStyleNet()
total = sum(p.numel() for p in vgg_model.parameters())
print(f"VGG 風格模型參數量: {total:,}")

dummy = torch.rand(1, 3, 32, 32)
out = vgg_model(dummy)
print(f"輸入: {dummy.shape} → 輸出: {out.shape}")


# ─────────────────────────────────────────────────────────────
# 5.7 練習題
# ─────────────────────────────────────────────────────────────
print("\n\n📌 5.7 練習題")
print("-" * 40)
print("""
練習 1：嘗試不同的資料增強組合：
        - 加入 transforms.RandomRotation(15)
        - 加入 transforms.RandomErasing()
        比較對測試準確率的影響

練習 2：修改 CNN 架構：
        - 增加一個卷積區塊（第 4 層）
        - 使用不同的 kernel_size（5x5 vs 3x3）
        觀察參數量和準確率的變化

練習 3：實作 Learning Rate Finder：
        - 從很小的學習率開始（1e-7），逐漸增大
        - 記錄每個學習率對應的 loss
        - 找出 loss 開始下降最快的學習率

練習 4：在 CIFAR-10 上使用 MNIST 的 SimpleCNN（第三章）
        觀察為什麼淺層 CNN 在更複雜的資料集上表現較差
""")

print("\n" + "=" * 60)
print("第五章結束！下一章：實戰 — NLP 文字分類")
print("=" * 60)

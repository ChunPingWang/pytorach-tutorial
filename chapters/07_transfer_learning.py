"""
=============================================================================
第七章：遷移學習（Transfer Learning）
=============================================================================

什麼是遷移學習？
──────────────
把「別人在大型資料集上訓練好的模型」拿來用在你的小型資料集上。

比喻：
  就像一個學過英語的人去學法語，比從零開始快很多！
  因為他已經懂了「語言」的基本概念（語法、詞彙結構...）

為什麼遷移學習有效？
──────────────────
1. 預訓練模型（如 ResNet）已經在 ImageNet 上學會了：
   - 淺層：邊緣、紋理、顏色（通用特徵）
   - 中層：形狀、圖案（半通用特徵）
   - 深層：物體部件（任務相關特徵）

2. 你的小資料集可能只有幾百張圖：
   - 從零訓練 → 嚴重過擬合
   - 用遷移學習 → 效果好很多！

兩種遷移學習策略：
──────────────────

策略一：Feature Extraction（特徵提取）
  凍結所有預訓練層，只訓練最後的分類頭
  適用：資料量很少（< 1000 張）
  ┌─────────────────────────┐
  │ 預訓練層（凍結，不更新）   │ ← 不訓練
  ├─────────────────────────┤
  │ 新的分類層               │ ← 只訓練這裡
  └─────────────────────────┘

策略二：Fine-tuning（微調）
  先凍結預訓練層，訓練分類頭
  再解凍部分（或全部）預訓練層，用小學習率微調
  適用：資料量中等（1000~10000 張）
  ┌─────────────────────────┐
  │ 預訓練淺層（凍結）        │ ← 不訓練
  ├─────────────────────────┤
  │ 預訓練深層（解凍，小 LR）  │ ← 用小學習率微調
  ├─────────────────────────┤
  │ 新的分類層               │ ← 用正常學習率訓練
  └─────────────────────────┘

本章學習目標：
─────────────
✓ 載入預訓練模型
✓ 凍結/解凍參數
✓ 替換分類頭
✓ 差異化學習率
✓ 實戰：用 ResNet18 做花卉分類
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

print("=" * 60)
print("第七章：遷移學習（Transfer Learning）")
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
# 7.1 載入預訓練模型
# ─────────────────────────────────────────────────────────────
print("\n📌 7.1 載入預訓練模型")
print("-" * 40)

# PyTorch 提供的預訓練模型（在 ImageNet 上訓練）
# ImageNet: 1000 類、超過 100 萬張圖片
print("""
  常用預訓練模型：
  ┌──────────────┬────────────┬───────────┬──────────────┐
  │ 模型          │ 參數量      │ Top-1 Acc │ 適用場景      │
  ├──────────────┼────────────┼───────────┼──────────────┤
  │ ResNet-18    │ 11.7M      │ 69.8%     │ 快速原型       │
  │ ResNet-50    │ 25.6M      │ 76.1%     │ 一般任務       │
  │ EfficientNet │ 5.3M~66M  │ 77~84%    │ 效率優先       │
  │ ViT          │ 86M~632M  │ 81~88%    │ 大資料集       │
  └──────────────┴────────────┴───────────┴──────────────┘
""")

# 載入 ResNet18（預訓練版本）
try:
    # PyTorch 2.0+ 的新寫法
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    print("已載入預訓練 ResNet18（新 API）")
except Exception:
    try:
        # 舊版相容寫法
        resnet18 = models.resnet18(pretrained=True)
        print("已載入預訓練 ResNet18（舊 API）")
    except Exception:
        # 無法下載時，用隨機初始化
        resnet18 = models.resnet18(pretrained=False)
        print("無法下載預訓練權重，使用隨機初始化")

# 看看 ResNet18 的最後幾層
print(f"\nResNet18 最後的全連接層: {resnet18.fc}")
# Linear(in_features=512, out_features=1000)  ← 1000 類（ImageNet）

# 查看所有層的名稱
print("\nResNet18 的主要層：")
for name, module in resnet18.named_children():
    if isinstance(module, nn.Sequential):
        print(f"  {name}: Sequential with {len(module)} blocks")
    else:
        print(f"  {name}: {module.__class__.__name__}")


# ─────────────────────────────────────────────────────────────
# 7.2 策略一：Feature Extraction（凍結 + 替換分類頭）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 7.2 Feature Extraction（特徵提取）")
print("-" * 40)

def create_feature_extractor(num_classes):
    """
    用預訓練 ResNet18 做特徵提取

    步驟：
    1. 載入預訓練模型
    2. 凍結所有參數（requires_grad = False）
    3. 替換最後的分類層
    """
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=False)

    # 步驟一：凍結所有參數
    for param in model.parameters():
        param.requires_grad = False

    # 步驟二：替換分類層（新的層預設 requires_grad=True）
    num_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model

# 建立 5 類花卉分類器
model_fe = create_feature_extractor(num_classes=5)

# 確認：只有新的分類層需要訓練
trainable = sum(p.numel() for p in model_fe.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_fe.parameters())
print(f"總參數量: {total:,}")
print(f"可訓練參數: {trainable:,}")
print(f"凍結參數: {total - trainable:,}")
print(f"可訓練比例: {trainable/total*100:.1f}%")

# 只把需要訓練的參數傳給優化器
optimizer_fe = optim.Adam(model_fe.fc.parameters(), lr=0.001)
print(f"\n只優化分類層的參數")


# ─────────────────────────────────────────────────────────────
# 7.3 策略二：Fine-tuning（微調）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 7.3 Fine-tuning（微調）")
print("-" * 40)

def create_finetuned_model(num_classes, unfreeze_from='layer3'):
    """
    微調預訓練 ResNet18

    步驟：
    1. 載入預訓練模型
    2. 凍結淺層，解凍深層
    3. 替換分類層
    4. 對不同層使用不同學習率
    """
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=False)

    # 先凍結所有層
    for param in model.parameters():
        param.requires_grad = False

    # 解凍指定的層
    unfreeze_layers = {
        'layer4': [model.layer4],
        'layer3': [model.layer3, model.layer4],
        'layer2': [model.layer2, model.layer3, model.layer4],
        'all': [model.layer1, model.layer2, model.layer3, model.layer4],
    }

    for layer in unfreeze_layers.get(unfreeze_from, []):
        for param in layer.parameters():
            param.requires_grad = True

    # 替換分類層
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model

# 微調從 layer3 開始
model_ft = create_finetuned_model(num_classes=5, unfreeze_from='layer3')

trainable = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_ft.parameters())
print(f"Fine-tuning 從 layer3 開始：")
print(f"  可訓練: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

# 差異化學習率（Discriminative Learning Rates）
# 淺層用小學習率（微調），深層用大學習率（新學）
optimizer_ft = optim.Adam([
    {'params': model_ft.layer3.parameters(), 'lr': 1e-5},   # 淺層：小學習率
    {'params': model_ft.layer4.parameters(), 'lr': 1e-4},   # 深層：中學習率
    {'params': model_ft.fc.parameters(), 'lr': 1e-3},       # 分類頭：大學習率
])

print("\n差異化學習率：")
for i, group in enumerate(optimizer_ft.param_groups):
    num_params = sum(p.numel() for p in group['params'])
    print(f"  Group {i}: lr={group['lr']}, params={num_params:,}")


# ─────────────────────────────────────────────────────────────
# 7.4 實戰：花卉分類
# ─────────────────────────────────────────────────────────────
print("\n\n📌 7.4 實戰：花卉分類")
print("-" * 40)

# 模擬一個小型花卉資料集（真實場景中用 ImageFolder 載入）
# 5 類花卉：玫瑰、向日葵、蒲公英、鬱金香、雛菊
class FakeFlowerDataset(Dataset):
    """模擬花卉資料集（教學用）"""
    def __init__(self, num_samples=500, num_classes=5):
        self.data = torch.randn(num_samples, 3, 224, 224) * 0.5
        self.labels = torch.randint(0, num_classes, (num_samples,))
        # 讓不同類別有稍微不同的特徵分佈
        for i in range(num_classes):
            mask = self.labels == i
            self.data[mask] += (i * 0.1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 建立資料集
train_dataset = FakeFlowerDataset(400)
test_dataset = FakeFlowerDataset(100)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

flower_names = ['玫瑰', '向日葵', '蒲公英', '鬱金香', '雛菊']
print(f"訓練集: {len(train_dataset)} 張")
print(f"測試集: {len(test_dataset)} 張")
print(f"類別: {flower_names}")

# 使用 Feature Extraction 策略
print("\n使用 Feature Extraction 策略訓練：")
model = create_feature_extractor(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
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

    # 測試
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * test_correct / test_total
    print(f"  Epoch {epoch+1} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")


# ─────────────────────────────────────────────────────────────
# 7.5 使用 ImageFolder 載入自己的資料（真實場景）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 7.5 使用 ImageFolder 載入自己的資料")
print("-" * 40)

print("""
  當你有自己的圖片資料時，只需要按照以下結構整理：

  data/
  ├── train/
  │   ├── cat/          ← 類別名稱 = 資料夾名稱
  │   │   ├── 001.jpg
  │   │   ├── 002.jpg
  │   │   └── ...
  │   ├── dog/
  │   │   ├── 001.jpg
  │   │   └── ...
  │   └── bird/
  │       └── ...
  └── test/
      ├── cat/
      ├── dog/
      └── bird/

  程式碼：
  ```python
  from torchvision.datasets import ImageFolder

  train_transform = transforms.Compose([
      transforms.Resize(256),           # 先放大
      transforms.CenterCrop(224),       # 再裁成 224x224
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
  ])

  train_dataset = ImageFolder('data/train', transform=train_transform)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

  # 類別名稱自動從資料夾名稱取得
  print(train_dataset.classes)  # ['bird', 'cat', 'dog']
  ```
""")


# ─────────────────────────────────────────────────────────────
# 7.6 遷移學習最佳實踐
# ─────────────────────────────────────────────────────────────
print("\n📌 7.6 遷移學習最佳實踐")
print("-" * 40)

print("""
  1. 資料前處理：
     - 使用預訓練模型相同的 Normalize 數值
     - ImageNet 的標準值：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
     - 圖片大小通常 resize 到 224×224

  2. 策略選擇：
     ┌────────────────┬──────────────────────────────┐
     │ 資料量 < 1000   │ Feature Extraction（凍結所有）│
     │ 資料量 1000~10K │ Fine-tune 深層               │
     │ 資料量 > 10K    │ Fine-tune 全部或從零訓練      │
     └────────────────┴──────────────────────────────┘

  3. 學習率設定：
     - 凍結層不需要學習率
     - 淺層（微調）：1e-5 ~ 1e-4
     - 深層（微調）：1e-4 ~ 1e-3
     - 分類頭（新的）：1e-3 ~ 1e-2

  4. 訓練技巧：
     - 先訓練分類頭 3~5 epochs
     - 再解凍深層，用小學習率微調 10~20 epochs
     - 使用 CosineAnnealing 或 ReduceLROnPlateau
     - 加入資料增強防止過擬合

  5. 避免的錯誤：
     - 忘記設定 model.eval() 進行測試
     - 忘記用相同的 Normalize 數值
     - 微調時學習率太大（破壞預訓練特徵）
""")


# ─────────────────────────────────────────────────────────────
# 7.7 練習題
# ─────────────────────────────────────────────────────────────
print("\n\n📌 7.7 練習題")
print("-" * 40)
print("""
練習 1：用 EfficientNet_B0 替換 ResNet18，比較效果
        model = models.efficientnet_b0(weights='DEFAULT')
        model.classifier[1] = nn.Linear(1280, num_classes)

練習 2：實作「漸進式解凍」：
        - 先只訓練分類頭 5 epochs
        - 解凍 layer4 再訓練 5 epochs
        - 解凍 layer3 再訓練 5 epochs
        觀察每階段的準確率變化

練習 3：用自己的圖片資料（至少 3 類，每類 20+ 張）：
        - 用 ImageFolder 載入
        - 用遷移學習訓練分類器
        - 匯出模型做推論
""")

print("\n" + "=" * 60)
print("第七章結束！下一章：實戰 — 生成對抗網路（GAN）")
print("=" * 60)

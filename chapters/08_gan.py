"""
=============================================================================
第八章：實戰 — 生成對抗網路（GAN）
=============================================================================

什麼是 GAN？
──────────
GAN（Generative Adversarial Network）= 生成對抗網路
由兩個神經網路互相對抗訓練：

  ┌──────────────┐         ┌──────────────┐
  │   Generator   │  生成   │ Discriminator │
  │   生成器       │ ────→  │  判別器       │
  │   (造假者)     │  假圖片  │  (鑑定師)     │
  └──────────────┘         └──────────────┘
        ↑                         │
        │    「這是真的還是假的？」    │
        └────── 回饋 ──────────────┘

比喻：
  生成器 = 偽鈔製造者（努力讓假鈔看起來像真的）
  判別器 = 銀行驗鈔員（努力分辨真鈔和假鈔）

兩者互相對抗，最終：
  - 生成器越來越會造假 → 生成的圖片越來越逼真
  - 判別器越來越會鑑別 → 逼迫生成器做得更好

訓練流程：
─────────
每個 iteration：
  1. 訓練判別器：
     - 給真圖片 → 判別器應該輸出「真」
     - 給假圖片 → 判別器應該輸出「假」

  2. 訓練生成器：
     - 生成假圖片 → 讓判別器誤認為「真」

本章學習目標：
─────────────
✓ GAN 的原理與訓練邏輯
✓ 生成器和判別器的設計
✓ GAN 的訓練技巧
✓ 實戰：生成手寫數字（MNIST）
✓ 常見問題：模式崩潰、訓練不穩定
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

print("=" * 60)
print("第八章：實戰 — 生成對抗網路（GAN）")
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
# 8.1 載入 MNIST 資料集
# ─────────────────────────────────────────────────────────────
print("\n📌 8.1 載入 MNIST 資料集")
print("-" * 40)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])   # 標準化到 [-1, 1]
])

try:
    mnist_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    DATA_AVAILABLE = True
except Exception:
    # 沒有網路時用假資料
    class FakeMNIST(torch.utils.data.Dataset):
        def __init__(self, size=10000):
            self.data = torch.randn(size, 1, 28, 28) * 0.5
            self.targets = torch.randint(0, 10, (size,))
        def __len__(self):
            return len(self.targets)
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    mnist_dataset = FakeMNIST()
    DATA_AVAILABLE = False

dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True, num_workers=0)
print(f"資料集大小: {len(mnist_dataset)}")
print(f"圖片形狀: (1, 28, 28) — 灰階手寫數字")


# ─────────────────────────────────────────────────────────────
# 8.2 定義生成器（Generator）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 8.2 定義生成器")
print("-" * 40)

# 潛在空間維度（latent space）
# 生成器的輸入：一個隨機雜訊向量
LATENT_DIM = 100

class Generator(nn.Module):
    """
    生成器：把隨機雜訊 → 逼真的圖片

    結構：
    噪音(100) → FC → Reshape → ConvT → ConvT → ConvT → 圖片(1,28,28)

    ConvTranspose2d（轉置卷積）：
    - 普通卷積：大圖 → 小圖（提取特徵）
    - 轉置卷積：小圖 → 大圖（生成圖片）
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            # (256, 7, 7) → (128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # (128, 14, 14) → (64, 28, 28)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (64, 28, 28) → (1, 28, 28)
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()   # 輸出範圍 [-1, 1]（配合資料的標準化）
        )

    def forward(self, z):
        # z: (batch, latent_dim)
        x = self.fc(z)                          # (batch, 256*7*7)
        x = x.view(-1, 256, 7, 7)              # (batch, 256, 7, 7)
        x = self.conv(x)                        # (batch, 1, 28, 28)
        return x


# ─────────────────────────────────────────────────────────────
# 8.3 定義判別器（Discriminator）
# ─────────────────────────────────────────────────────────────
print("\n📌 8.3 定義判別器")
print("-" * 40)

class Discriminator(nn.Module):
    """
    判別器：判斷圖片是「真」還是「假」

    結構：
    圖片(1,28,28) → Conv → Conv → Conv → Flatten → FC → 真/假

    LeakyReLU 而非 ReLU：
    - 在 GAN 中，LeakyReLU 對判別器的表現更好
    - 避免 ReLU 的「dead neuron」問題
    """
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            # (1, 28, 28) → (64, 14, 14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # (64, 14, 14) → (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # (128, 7, 7) → (256, 3, 3)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1)
            # 不加 Sigmoid！因為我們用 BCEWithLogitsLoss
        )

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        features = self.conv(x)          # (batch, 256, 3, 3)
        output = self.fc(features)       # (batch, 1)
        return output.squeeze(1)         # (batch,)


# 建立模型
generator = Generator().to(device)
discriminator = Discriminator().to(device)

print(f"生成器參數量: {sum(p.numel() for p in generator.parameters()):,}")
print(f"判別器參數量: {sum(p.numel() for p in discriminator.parameters()):,}")

# 測試
z = torch.randn(4, LATENT_DIM).to(device)
fake_images = generator(z)
d_output = discriminator(fake_images)
print(f"\n生成器：噪音 {z.shape} → 圖片 {fake_images.shape}")
print(f"判別器：圖片 {fake_images.shape} → 分數 {d_output.shape}")


# ─────────────────────────────────────────────────────────────
# 8.4 訓練 GAN
# ─────────────────────────────────────────────────────────────
print("\n\n📌 8.4 訓練 GAN")
print("-" * 40)

# 損失函數和優化器
criterion = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 訓練標籤
# real = 1, fake = 0

num_epochs = 5  # 完整訓練需要 50+ epochs
print(f"開始訓練 GAN（{num_epochs} epochs）：\n")

for epoch in range(num_epochs):
    g_losses = []
    d_losses = []

    for batch_idx, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # 建立標籤
        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        # ─────────────────────────────────
        # 步驟一：訓練判別器
        # ─────────────────────────────────
        # 目標：讓判別器正確分辨真假

        optimizer_D.zero_grad()

        # 1a. 用真圖片訓練 → 判別器應輸出「真」
        d_real_output = discriminator(real_images)
        d_real_loss = criterion(d_real_output, real_labels)

        # 1b. 用假圖片訓練 → 判別器應輸出「假」
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_images = generator(z)
        d_fake_output = discriminator(fake_images.detach())  # detach！不要更新生成器
        d_fake_loss = criterion(d_fake_output, fake_labels)

        # 判別器總損失
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        # ─────────────────────────────────
        # 步驟二：訓練生成器
        # ─────────────────────────────────
        # 目標：讓生成器騙過判別器

        optimizer_G.zero_grad()

        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_images = generator(z)
        g_output = discriminator(fake_images)

        # 生成器想讓判別器認為假圖是真的
        g_loss = criterion(g_output, real_labels)  # 用 real_labels！
        g_loss.backward()
        optimizer_G.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    # 每個 epoch 的統計
    avg_d_loss = sum(d_losses) / len(d_losses)
    avg_g_loss = sum(g_losses) / len(g_losses)
    print(f"  Epoch {epoch+1}/{num_epochs} | "
          f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

    # 每個 epoch 生成樣本展示
    generator.eval()
    with torch.no_grad():
        z = torch.randn(8, LATENT_DIM).to(device)
        samples = generator(z)
        # 統計生成圖片的像素值範圍
        print(f"    生成圖片像素範圍: [{samples.min().item():.2f}, {samples.max().item():.2f}]")
    generator.train()


# ─────────────────────────────────────────────────────────────
# 8.5 生成圖片
# ─────────────────────────────────────────────────────────────
print("\n\n📌 8.5 生成圖片")
print("-" * 40)

generator.eval()
with torch.no_grad():
    # 生成 16 張圖片
    z = torch.randn(16, LATENT_DIM).to(device)
    generated = generator(z).cpu()

    print(f"生成了 {generated.shape[0]} 張 {generated.shape[2]}x{generated.shape[3]} 的圖片")
    print(f"像素值範圍: [{generated.min():.3f}, {generated.max():.3f}]")

    # 在真實應用中，你可以用 torchvision.utils.save_image 儲存
    # torchvision.utils.save_image(generated, 'generated.png', nrow=4, normalize=True)
    print("\n（完整訓練 50+ epochs 後，生成的數字會越來越清晰）")


# ─────────────────────────────────────────────────────────────
# 8.6 GAN 訓練技巧與常見問題
# ─────────────────────────────────────────────────────────────
print("\n\n📌 8.6 GAN 訓練技巧與常見問題")
print("-" * 40)

print("""
  常見問題：

  1. 模式崩潰（Mode Collapse）
     症狀：生成器只生成少數幾種圖片
     原因：生成器找到了一種能騙過判別器的「捷徑」
     解法：
     - 使用 Wasserstein GAN (WGAN)
     - Label Smoothing（真標籤用 0.9 而非 1.0）
     - 加入雜訊到判別器的輸入

  2. 訓練不穩定（判別器太強或太弱）
     症狀：D Loss 趨近 0（判別器太強）或 G Loss 不下降
     解法：
     - 調整 D 和 G 的訓練比例
     - 使用 spectral normalization
     - 用 WGAN-GP（gradient penalty）

  3. 梯度消失
     症狀：G Loss 不下降
     原因：判別器太厲害，生成器得不到有效梯度
     解法：
     - 用 LSGAN（最小二乘 GAN）
     - 用 feature matching loss

  訓練技巧：
  ┌────────────────────────────────────────────┐
  │ 1. 使用 Adam (lr=0.0002, betas=(0.5, 0.999)) │
  │ 2. 判別器用 LeakyReLU (0.2)                  │
  │ 3. 生成器用 ReLU + 最後一層 Tanh              │
  │ 4. 使用 BatchNorm（判別器第一層除外）          │
  │ 5. 標籤平滑：real=0.9, fake=0.0              │
  │ 6. 每隔幾步才訓練判別器或生成器               │
  └────────────────────────────────────────────┘
""")

# 標籤平滑的實作
print("標籤平滑範例：")
real_labels_smooth = torch.FloatTensor(64).uniform_(0.8, 1.0)
print(f"  平滑的真標籤: {real_labels_smooth[:5].tolist()}")


# ─────────────────────────────────────────────────────────────
# 8.7 進階：條件式 GAN（Conditional GAN）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 8.7 進階：條件式 GAN（CGAN）")
print("-" * 40)

print("""
  普通 GAN：隨機生成 → 無法控制生成什麼
  條件 GAN：指定條件 → 生成特定類別的圖片

  例如：指定「3」→ 生成手寫數字 3

  做法：把類別標籤也餵給生成器和判別器
""")

class ConditionalGenerator(nn.Module):
    """條件式生成器：指定要生成哪個數字"""
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()

        # 把類別標籤轉成向量
        self.label_embedding = nn.Embedding(num_classes, 10)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 10, 256 * 7 * 7),  # 噪音 + 標籤
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_vec = self.label_embedding(labels)    # (batch, 10)
        x = torch.cat([z, label_vec], dim=1)        # (batch, 110)
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        return self.conv(x)


cgan = ConditionalGenerator().to(device)
z = torch.randn(4, 100).to(device)
labels = torch.tensor([0, 3, 5, 7]).to(device)  # 想生成 0, 3, 5, 7
generated = cgan(z, labels)
print(f"條件式生成：指定 {labels.cpu().tolist()} → 輸出 {generated.shape}")


# ─────────────────────────────────────────────────────────────
# 8.8 練習題
# ─────────────────────────────────────────────────────────────
print("\n\n📌 8.8 練習題")
print("-" * 40)
print("""
練習 1：增加訓練 epochs 到 50，觀察生成品質的變化
        每 10 epochs 儲存一次生成的圖片

練習 2：實作 WGAN-GP（Wasserstein GAN with Gradient Penalty）
        提示：
        - 判別器不用 Sigmoid
        - Loss = D(real) - D(fake) + gradient_penalty
        - gradient_penalty = (||grad(D(interpolated))||_2 - 1)²

練習 3：完成 Conditional GAN 的完整訓練
        - 加入 ConditionalDiscriminator
        - 訓練後，指定生成 0~9 每個數字

練習 4：在 Fashion-MNIST 上訓練 GAN
        生成衣服、鞋子等時尚商品的圖片
""")

print("\n" + "=" * 60)
print("第八章結束！下一章：模型部署")
print("=" * 60)

"""
=============================================================================
第一章：PyTorch Tensor 張量基礎
=============================================================================

什麼是 Tensor（張量）？
─────────────────────
Tensor 是 PyTorch 中最核心的資料結構，你可以把它想像成：
- 0 維 Tensor = 一個數字（純量 scalar）       例如：溫度 36.5°C
- 1 維 Tensor = 一排數字（向量 vector）        例如：[身高, 體重, 年齡]
- 2 維 Tensor = 一個表格（矩陣 matrix）        例如：Excel 試算表
- 3 維 Tensor = 一疊表格                       例如：彩色圖片 (高x寬x顏色)
- 4 維 Tensor = 一批圖片                       例如：一個 batch 的訓練資料

圖解：
    純量        向量          矩陣              3D Tensor
     5        [1,2,3]     [[1,2,3],        [[[1,2],[3,4]],
                           [4,5,6]]          [[5,6],[7,8]]]

為什麼不用 NumPy 就好？
─────────────────────
1. GPU 加速：Tensor 可以搬到 GPU 上運算，速度快 10~100 倍
2. 自動微分：Tensor 能自動計算梯度（訓練神經網路的關鍵）
3. 深度學習生態系：與 PyTorch 的神經網路模組無縫整合

本章學習目標：
─────────────
✓ 建立各種 Tensor
✓ Tensor 的基本運算（加減乘除、矩陣乘法）
✓ 索引與切片
✓ 形狀操作（reshape, view, squeeze）
✓ Tensor 與 NumPy 的互轉
✓ GPU 加速
=============================================================================
"""

import torch
import numpy as np

print("=" * 60)
print("第一章：PyTorch Tensor 張量基礎")
print(f"PyTorch 版本：{torch.__version__}")
print(f"CUDA 可用：{torch.cuda.is_available()}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1.1 建立 Tensor 的各種方式
# ─────────────────────────────────────────────────────────────
print("\n📌 1.1 建立 Tensor 的各種方式")
print("-" * 40)

# 方式一：從 Python list 建立
# 最直覺的方式，把 Python 的 list 直接轉成 Tensor
data_list = [1.0, 2.0, 3.0, 4.0, 5.0]
tensor_from_list = torch.tensor(data_list)
print(f"從 list 建立：{tensor_from_list}")
print(f"  資料型別：{tensor_from_list.dtype}")   # float32（預設）
print(f"  形狀：{tensor_from_list.shape}")        # torch.Size([5])
print(f"  維度數：{tensor_from_list.ndim}")        # 1

# 方式二：從 NumPy array 建立
# 如果你已經有 NumPy 的資料，可以直接轉換
np_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor_from_numpy = torch.from_numpy(np_array)
print(f"\n從 NumPy 建立：\n{tensor_from_numpy}")
print(f"  形狀：{tensor_from_numpy.shape}")       # torch.Size([2, 3])

# 重要！from_numpy 會共享記憶體（修改一個，另一個也會變）
np_array[0, 0] = 999
print(f"  修改 NumPy 後 Tensor 也變了：{tensor_from_numpy[0, 0]}")  # 999

# 如果不想共享記憶體，用 torch.tensor() 複製一份
tensor_copy = torch.tensor(np_array)  # 這是獨立的副本

# 方式三：建立特殊 Tensor
print("\n特殊 Tensor：")
zeros = torch.zeros(2, 3)           # 全零矩陣
ones = torch.ones(2, 3)             # 全一矩陣
rand = torch.rand(2, 3)             # 均勻分佈隨機 [0, 1)
randn = torch.randn(2, 3)           # 常態分佈隨機 (mean=0, std=1)
arange = torch.arange(0, 10, 2)     # 等差數列 [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # 等間距 [0, 0.25, 0.5, 0.75, 1]
eye = torch.eye(3)                  # 單位矩陣

print(f"  zeros(2,3):\n{zeros}")
print(f"  ones(2,3):\n{ones}")
print(f"  rand(2,3):\n{rand}")
print(f"  arange(0,10,2): {arange}")
print(f"  linspace(0,1,5): {linspace}")
print(f"  eye(3):\n{eye}")

# 方式四：建立與現有 Tensor 相同形狀的新 Tensor
# 在實際開發中非常常用
template = torch.rand(3, 4)
zeros_like = torch.zeros_like(template)   # 相同形狀的全零 Tensor
ones_like = torch.ones_like(template)     # 相同形狀的全一 Tensor
rand_like = torch.rand_like(template)     # 相同形狀的隨機 Tensor
print(f"\n  template 形狀：{template.shape}")
print(f"  zeros_like 形狀：{zeros_like.shape}")  # 一樣是 (3, 4)


# ─────────────────────────────────────────────────────────────
# 1.2 資料型別（dtype）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 1.2 資料型別（dtype）")
print("-" * 40)

# PyTorch 常用的資料型別：
# torch.float32 (預設) — 大部分神經網路使用
# torch.float64        — 需要高精度時使用
# torch.float16        — 混合精度訓練，節省記憶體
# torch.int32          — 整數索引
# torch.int64 (long)   — 分類標籤常用
# torch.bool           — 遮罩（mask）操作

# 指定型別建立
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
bool_tensor = torch.tensor([True, False, True])

print(f"int32: {int_tensor}, dtype={int_tensor.dtype}")
print(f"float32: {float_tensor}, dtype={float_tensor.dtype}")
print(f"bool: {bool_tensor}, dtype={bool_tensor.dtype}")

# 型別轉換
converted = int_tensor.float()          # int32 → float32
converted2 = float_tensor.long()        # float32 → int64
converted3 = float_tensor.to(torch.float16)  # 明確指定目標型別
print(f"\nint → float: {converted}, dtype={converted.dtype}")
print(f"float → long: {converted2}, dtype={converted2.dtype}")

# 實際案例：為什麼分類標籤要用 long (int64)？
# 因為 PyTorch 的 CrossEntropyLoss 要求標籤必須是 LongTensor
labels = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
print(f"\n分類標籤（必須是 long）：{labels}, dtype={labels.dtype}")


# ─────────────────────────────────────────────────────────────
# 1.3 基本運算
# ─────────────────────────────────────────────────────────────
print("\n\n📌 1.3 基本運算")
print("-" * 40)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 逐元素運算（element-wise）— 對應位置的元素分別運算
print("逐元素運算（element-wise）：")
print(f"  a + b = {a + b}")          # [5, 7, 9]
print(f"  a - b = {a - b}")          # [-3, -3, -3]
print(f"  a * b = {a * b}")          # [4, 10, 18]  注意：這不是矩陣乘法！
print(f"  a / b = {a / b}")          # [0.25, 0.4, 0.5]
print(f"  a ** 2 = {a ** 2}")        # [1, 4, 9]  次方

# 常用數學函數
print(f"\n  torch.sqrt(a) = {torch.sqrt(a)}")        # 開根號
print(f"  torch.exp(a) = {torch.exp(a)}")            # e 的次方
print(f"  torch.log(a) = {torch.log(a)}")            # 自然對數
print(f"  torch.abs(torch.tensor([-1,2,-3])) = {torch.abs(torch.tensor([-1,2,-3]))}")

# 統計運算
print(f"\n統計運算：")
data = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
print(f"  data = {data}")
print(f"  平均值 mean: {data.mean()}")         # 6.0
print(f"  總和 sum: {data.sum()}")              # 30.0
print(f"  最大值 max: {data.max()}")            # 10.0
print(f"  最小值 min: {data.min()}")            # 2.0
print(f"  標準差 std: {data.std()}")            # ≈ 3.16
print(f"  最大值索引 argmax: {data.argmax()}")  # 4（第 5 個元素）

# 矩陣運算
print(f"\n矩陣運算：")
mat_a = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]])
mat_b = torch.tensor([[5.0, 6.0],
                       [7.0, 8.0]])

# 矩陣乘法（三種等價寫法）
result1 = torch.matmul(mat_a, mat_b)    # 函數寫法
result2 = mat_a @ mat_b                 # 運算子寫法（推薦，最簡潔）
result3 = torch.mm(mat_a, mat_b)        # 僅限 2D 矩陣

print(f"  mat_a @ mat_b =\n{result1}")
# [[1*5+2*7, 1*6+2*8],   = [[19, 22],
#  [3*5+4*7, 3*6+4*8]]      [43, 50]]

# 轉置
print(f"\n  mat_a 轉置:\n{mat_a.T}")
print(f"  也可以用 mat_a.transpose(0,1):\n{mat_a.transpose(0, 1)}")


# ─────────────────────────────────────────────────────────────
# 1.4 索引與切片
# ─────────────────────────────────────────────────────────────
print("\n\n📌 1.4 索引與切片")
print("-" * 40)

# 建立一個 4x5 的矩陣來練習
matrix = torch.arange(20).reshape(4, 5).float()
print(f"原始矩陣 (4x5):\n{matrix}")

# 基本索引（跟 Python list / NumPy 一樣）
print(f"\n  matrix[0]     = {matrix[0]}")        # 第 0 列（整列）
print(f"  matrix[0, 2]  = {matrix[0, 2]}")      # 第 0 列第 2 行 = 2.0
print(f"  matrix[-1]    = {matrix[-1]}")         # 最後一列
print(f"  matrix[:, 0]  = {matrix[:, 0]}")       # 所有列的第 0 行（整行）

# 切片
print(f"\n  matrix[1:3]   =\n{matrix[1:3]}")     # 第 1~2 列
print(f"  matrix[:, 1:4] =\n{matrix[:, 1:4]}")  # 所有列的第 1~3 行

# 進階索引：布林遮罩（Boolean Mask）
# 實際場景：篩選大於某個值的元素
mask = matrix > 10
print(f"\n  布林遮罩 (matrix > 10):\n{mask}")
print(f"  篩選結果：{matrix[mask]}")    # 取出所有 > 10 的元素

# Fancy indexing（用 Tensor 當索引）
indices = torch.tensor([0, 2, 3])
print(f"\n  取出第 0, 2, 3 列:\n{matrix[indices]}")


# ─────────────────────────────────────────────────────────────
# 1.5 形狀操作（非常重要！）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 1.5 形狀操作")
print("-" * 40)

# 為什麼形狀操作很重要？
# 在深度學習中，不同的層對輸入形狀有特定要求：
# - 全連接層（Linear）需要 2D: (batch_size, features)
# - CNN 需要 4D: (batch_size, channels, height, width)
# - RNN 需要 3D: (sequence_length, batch_size, features)
# 你經常需要調整 Tensor 形狀來餵給不同的層

original = torch.arange(12).float()
print(f"原始 1D Tensor: {original}")
print(f"  形狀: {original.shape}")  # torch.Size([12])

# reshape：改變形狀（最常用）
reshaped = original.reshape(3, 4)
print(f"\nreshape(3, 4):\n{reshaped}")

# view：跟 reshape 類似，但要求記憶體連續
viewed = original.view(3, 4)
print(f"\nview(3, 4):\n{viewed}")

# 使用 -1 讓 PyTorch 自動計算該維度的大小
auto_shape = original.reshape(2, -1)  # 2 x ? → 2 x 6
print(f"\nreshape(2, -1) 自動計算: {auto_shape.shape}")  # (2, 6)

# unsqueeze：增加一個維度（常用於把資料餵給 CNN）
# 例如：一張灰階圖片是 (28, 28)，CNN 需要 (1, 1, 28, 28)
img = torch.rand(28, 28)
print(f"\n原始圖片形狀: {img.shape}")              # [28, 28]
img_batch = img.unsqueeze(0).unsqueeze(0)
print(f"加了 batch 和 channel: {img_batch.shape}")  # [1, 1, 28, 28]

# squeeze：移除大小為 1 的維度
squeezed = img_batch.squeeze()
print(f"squeeze 後: {squeezed.shape}")              # [28, 28]

# permute：重新排列維度順序
# 實際場景：圖片從 (H, W, C) 轉成 PyTorch 要的 (C, H, W)
hwc_image = torch.rand(224, 224, 3)    # H=224, W=224, C=3 (RGB)
chw_image = hwc_image.permute(2, 0, 1)  # 把 C 移到最前面
print(f"\nHWC 圖片: {hwc_image.shape} → CHW 圖片: {chw_image.shape}")

# flatten：攤平成 1D（CNN 輸出接全連接層時常用）
batch_features = torch.rand(32, 64, 7, 7)  # (batch, channels, h, w)
flat = batch_features.flatten(1)            # 從第 1 維開始攤平，保留 batch
print(f"\nflatten: {batch_features.shape} → {flat.shape}")  # (32, 3136)

# cat：串接 Tensor（合併資料）
t1 = torch.tensor([[1, 2], [3, 4]])
t2 = torch.tensor([[5, 6], [7, 8]])
cat_row = torch.cat([t1, t2], dim=0)     # 沿列方向串接
cat_col = torch.cat([t1, t2], dim=1)     # 沿行方向串接
print(f"\ncat dim=0:\n{cat_row}")   # shape: (4, 2)
print(f"cat dim=1:\n{cat_col}")     # shape: (2, 4)

# stack：堆疊 Tensor（新增一個維度）
stacked = torch.stack([t1, t2], dim=0)
print(f"\nstack dim=0:\n{stacked}")        # shape: (2, 2, 2)
print(f"  形狀: {stacked.shape}")


# ─────────────────────────────────────────────────────────────
# 1.6 Broadcasting（廣播機制）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 1.6 Broadcasting（廣播機制）")
print("-" * 40)

# 當兩個 Tensor 形狀不同時，PyTorch 會自動「廣播」較小的 Tensor
# 規則：從最後一個維度開始比較
# - 維度大小相同 → OK
# - 其中一個是 1 → 自動擴展
# - 兩個都不是 1 且不相同 → 錯誤！

# 實際案例：對每一列減去平均值（標準化）
scores = torch.tensor([[80.0, 90.0, 70.0],
                        [60.0, 85.0, 95.0]])
print(f"原始成績:\n{scores}")

# 計算每一列的平均
row_mean = scores.mean(dim=1, keepdim=True)  # keepdim 保持維度
print(f"每列平均:\n{row_mean}")   # shape: (2, 1)

# 廣播：(2, 3) - (2, 1) → (2, 1) 自動擴展成 (2, 3)
normalized = scores - row_mean
print(f"標準化後:\n{normalized}")

# 另一個實際案例：給圖片的每個 channel 乘以不同權重
image = torch.rand(3, 224, 224)           # (C, H, W)
channel_weights = torch.tensor([0.3, 0.5, 0.2]).reshape(3, 1, 1)  # (3, 1, 1)
weighted = image * channel_weights         # (3,224,224) * (3,1,1) 自動廣播
print(f"\n圖片加權: {image.shape} * {channel_weights.shape} → {weighted.shape}")


# ─────────────────────────────────────────────────────────────
# 1.7 GPU 加速
# ─────────────────────────────────────────────────────────────
print("\n\n📌 1.7 GPU 加速")
print("-" * 40)

# PyTorch 有兩種 GPU 後端，取決於你的硬體：
#
#   後端    硬體                      怎麼檢查
#   ─────  ───────────────────────   ──────────────────────────────────
#   cuda    NVIDIA 顯示卡             torch.cuda.is_available()
#           （Colab、Windows、Linux）
#   mps     Apple Silicon             torch.backends.mps.is_available()
#           （M1/M2/M3/M4/M5 Mac）
#   cpu     都沒有的話                 永遠可用，只是比較慢
#
# 注意：Mac 上沒有 CUDA！寫 torch.cuda.is_available() 在 Mac 永遠是 False，
#      要用 MPS 才吃得到 Apple Silicon 的 GPU。

print(f"CUDA（NVIDIA GPU）可用：{torch.cuda.is_available()}")
print(f"MPS（Apple Silicon GPU）可用：{torch.backends.mps.is_available()}")


# 最佳實踐：寫一個能自動偵測裝置的函式，同一份程式碼到哪都能跑
def get_device():
    """自動選擇運算裝置：NVIDIA CUDA → Apple Silicon MPS → CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")      # NVIDIA GPU（Colab / Windows / Linux）
    if torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon GPU（M 系列 Mac）
    return torch.device("cpu")           # 都沒有就用 CPU，一樣跑得動


device = get_device()
print(f"\n自動選擇裝置: {device}")

# 印出裝置名稱（只有 CUDA 查得到型號）
if device.type == "cuda":
    print(f"  GPU 型號: {torch.cuda.get_device_name(0)}")
elif device.type == "mps":
    print("  使用 Apple Silicon 內建 GPU")
else:
    print("  沒有 GPU，所有操作一樣可以執行，只是較慢")

# 方法一：建立時直接放到指定裝置
gpu_tensor = torch.rand(1000, 1000, device=device)

# 方法二：從 CPU 搬到指定裝置
cpu_tensor = torch.rand(1000, 1000)
gpu_tensor2 = cpu_tensor.to(device)
# 也可以寫成 cpu_tensor.cuda()，但那樣就只能在 NVIDIA GPU 上跑了，不建議

# 在該裝置上運算
result = gpu_tensor @ gpu_tensor2

# 搬回 CPU（要轉成 NumPy 或給其他函式庫用時需要）
cpu_result = result.cpu()
print(f"  計算完成，結果形狀: {cpu_result.shape}")

# 注意：不同裝置上的 Tensor 不能直接運算！
# torch.add(cpu_tensor, gpu_tensor)  # 裝置不同會報錯！

data = torch.rand(100, 100, device=device)  # 自動放到正確的裝置上

# MPS 小提醒：少數算子還沒支援 MPS，遇到 NotImplementedError 時可以設環境變數
# 讓它自動退回 CPU 執行：
#   export PYTORCH_ENABLE_MPS_FALLBACK=1


# ─────────────────────────────────────────────────────────────
# 1.8 Tensor 與 NumPy 互轉
# ─────────────────────────────────────────────────────────────
print("\n\n📌 1.8 Tensor 與 NumPy 互轉")
print("-" * 40)

# Tensor → NumPy
tensor = torch.tensor([1.0, 2.0, 3.0])
np_from_tensor = tensor.numpy()     # 共享記憶體
print(f"Tensor → NumPy: {np_from_tensor}, type={type(np_from_tensor)}")

# 如果 Tensor 在 GPU 上，要先搬回 CPU
# gpu_tensor.cpu().numpy()

# 如果 Tensor 有 requires_grad=True，要先 detach
grad_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
# grad_tensor.numpy()  # 這會報錯！
np_from_grad = grad_tensor.detach().numpy()  # 正確做法
print(f"有梯度的 Tensor → NumPy: {np_from_grad}")

# NumPy → Tensor
np_arr = np.array([4.0, 5.0, 6.0])
tensor_from_np = torch.from_numpy(np_arr)    # 共享記憶體
tensor_copy = torch.tensor(np_arr)           # 複製一份（不共享）
print(f"NumPy → Tensor: {tensor_from_np}")


# ─────────────────────────────────────────────────────────────
# 1.9 實際案例：手動實作線性迴歸
# ─────────────────────────────────────────────────────────────
print("\n\n📌 1.9 實際案例：用 Tensor 手動實作線性迴歸")
print("-" * 40)

# 問題：已知房屋面積，預測房價
# 模型：price = weight * area + bias

# 產生模擬資料
torch.manual_seed(42)  # 固定隨機種子，確保結果可重現
areas = torch.rand(100) * 100            # 100 間房子，面積 0~100 坪
true_weight = 5.0                        # 真實每坪價格 5 萬
true_bias = 200.0                        # 基本價 200 萬
prices = true_weight * areas + true_bias + torch.randn(100) * 10  # 加入雜訊

print(f"資料範例 — 面積: {areas[:5].tolist()}")
print(f"資料範例 — 房價: {prices[:5].tolist()}")

# 初始化參數（隨機猜一個起點）
weight = torch.randn(1, requires_grad=False)  # 先不用 autograd
bias = torch.randn(1, requires_grad=False)
learning_rate = 0.0001

# 梯度下降訓練（純 Tensor 運算，不用 autograd）
print("\n開始訓練：")
for epoch in range(100):
    # 前向傳播：計算預測值
    predictions = weight * areas + bias

    # 計算損失（均方誤差 MSE）
    loss = ((predictions - prices) ** 2).mean()

    # 手動計算梯度
    # d(MSE)/d(weight) = 2/N * sum((pred - target) * area)
    # d(MSE)/d(bias) = 2/N * sum(pred - target)
    error = predictions - prices
    grad_weight = (2.0 / len(areas)) * (error * areas).sum()
    grad_bias = (2.0 / len(areas)) * error.sum()

    # 更新參數
    weight = weight - learning_rate * grad_weight
    bias = bias - learning_rate * grad_bias

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {loss.item():.2f} "
              f"| Weight: {weight.item():.3f} | Bias: {bias.item():.3f}")

print(f"\n訓練結果：weight={weight.item():.3f}（真實值=5.0）"
      f", bias={bias.item():.3f}（真實值=200.0）")


# ─────────────────────────────────────────────────────────────
# 1.10 練習題
# ─────────────────────────────────────────────────────────────
print("\n\n📌 1.10 練習題")
print("-" * 40)
print("""
練習 1：建立一個 3x3 的隨機矩陣，計算它的行列式（torch.linalg.det）
練習 2：建立兩個 (2,3) 矩陣，分別用 + 和 torch.add 相加，確認結果相同
練習 3：建立一個 24 元素的 1D Tensor，分別 reshape 成 (2,3,4) 和 (4,6)
練習 4：建立一個 (3,4) 矩陣，找出每一行的最大值及其索引
練習 5：模擬 5 個學生 3 科成績，用 broadcasting 計算每科的 z-score

提示：z-score = (x - mean) / std
""")

# 練習 4 的參考答案：
practice_matrix = torch.rand(3, 4) * 100
max_values, max_indices = practice_matrix.max(dim=1)
print(f"練習 4 參考：")
print(f"  矩陣:\n{practice_matrix}")
print(f"  每行最大值: {max_values}")
print(f"  最大值索引: {max_indices}")

print("\n\n" + "=" * 60)
print("第一章結束！下一章：自動微分 Autograd")
print("=" * 60)

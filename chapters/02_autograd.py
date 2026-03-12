"""
=============================================================================
第二章：自動微分 Autograd
=============================================================================

為什麼需要自動微分？
──────────────────
深度學習的核心公式：
    新參數 = 舊參數 - 學習率 × 梯度

「梯度」就是告訴我們「參數要往哪個方向調整，才能讓損失變小」。

手動計算梯度的問題：
- 簡單的 y = wx + b 還可以手算
- 但真實的神經網路有數百萬個參數，手算不可能
- PyTorch 的 Autograd 能自動幫你算所有梯度！

Autograd 的原理（計算圖）：
──────────────────────────

假設 y = (x + 2)² ，x = 3

前向傳播（計算結果）：
    x=3 → [+2] → a=5 → [²] → y=25

反向傳播（計算梯度）：
    dy/dx = dy/da × da/dx = 2a × 1 = 2×5 = 10

    y=25 ← [dy/da=2a=10] ← a=5 ← [da/dx=1] ← x=3

    所以 dy/dx = 10

PyTorch 會自動建立這個計算圖，然後用反向傳播算出所有梯度。

本章學習目標：
─────────────
✓ 理解 requires_grad 的作用
✓ 前向傳播與反向傳播
✓ 梯度的累加特性
✓ 停止梯度追蹤的方法
✓ 用 Autograd 重新實作線性迴歸
=============================================================================
"""

import torch

print("=" * 60)
print("第二章：自動微分 Autograd")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 2.1 requires_grad：開啟梯度追蹤
# ─────────────────────────────────────────────────────────────
print("\n📌 2.1 requires_grad：開啟梯度追蹤")
print("-" * 40)

# requires_grad=True 告訴 PyTorch：「請追蹤這個 Tensor 的所有運算」
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad = {x.requires_grad}")
print(f"x.grad = {x.grad}")  # 還沒算梯度，所以是 None

# 進行一些運算 → PyTorch 自動建立計算圖
y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1
print(f"\ny = x² + 2x + 1 = {y}")
print(f"y.grad_fn = {y.grad_fn}")  # 記錄了 y 是怎麼被計算出來的

# 反向傳播：計算 dy/dx
y.backward()

# 梯度結果
# dy/dx = 2x + 2 = 2(3) + 2 = 8
print(f"\n反向傳播後：")
print(f"x.grad = {x.grad}")  # tensor(8.)  ← dy/dx 在 x=3 的值


# ─────────────────────────────────────────────────────────────
# 2.2 計算圖詳解
# ─────────────────────────────────────────────────────────────
print("\n\n📌 2.2 計算圖詳解")
print("-" * 40)

# 更複雜的例子
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# 前向傳播：建立計算圖
#
#  a=2 ─→ [*] ─→ c=6 ─┐
#  b=3 ─┘               ├→ [+] ─→ e=11
#  b=3 ─→ [²] ─→ d=5 ──┘       (5 = b² - 4 不對)
#
# 更正：
c = a * b        # c = 2 * 3 = 6
d = b ** 2 - 4   # d = 9 - 4 = 5
e = c + d        # e = 6 + 5 = 11

print(f"a = {a.item()}, b = {b.item()}")
print(f"c = a * b = {c.item()}")
print(f"d = b² - 4 = {d.item()}")
print(f"e = c + d = {e.item()}")

# 反向傳播
e.backward()

# 手動驗證：
# e = a*b + b² - 4
# de/da = b = 3          ✓
# de/db = a + 2b = 2 + 6 = 8  ✓
print(f"\nde/da = {a.grad.item()} (應該是 {b.item()})")
print(f"de/db = {b.grad.item()} (應該是 {a.item() + 2*b.item()})")


# ─────────────────────────────────────────────────────────────
# 2.3 梯度累加（重要陷阱！）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 2.3 梯度累加（重要陷阱！）")
print("-" * 40)

# PyTorch 的梯度是「累加」的，不會自動清零！
# 這是初學者最常犯的錯誤之一

w = torch.tensor(1.0, requires_grad=True)

# 第一次計算
y1 = w * 3
y1.backward()
print(f"第一次 backward 後 w.grad = {w.grad}")  # 3.0

# 第二次計算（梯度會累加！）
y2 = w * 5
y2.backward()
print(f"第二次 backward 後 w.grad = {w.grad}")  # 3.0 + 5.0 = 8.0 !!!

# 正確做法：每次 backward 前要清零
w.grad.zero_()  # 就地清零（注意底線 _ 表示就地操作）
y3 = w * 7
y3.backward()
print(f"清零後重新計算 w.grad = {w.grad}")  # 7.0 ✓

print("""
  重要：在訓練迴圈中，每個 iteration 都要呼叫 optimizer.zero_grad()
  來清除上一步的梯度，否則梯度會越累積越大，訓練會爆掉！
""")


# ─────────────────────────────────────────────────────────────
# 2.4 向量/矩陣的梯度
# ─────────────────────────────────────────────────────────────
print("\n📌 2.4 向量/矩陣的梯度")
print("-" * 40)

# backward() 只能對純量（scalar）呼叫
# 如果結果是向量，需要先 .sum() 或指定 gradient 參數

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # y = [1, 4, 9]  ← 這是向量，不能直接 backward

# 方法一：先取總和，再 backward
loss = y.sum()  # 1 + 4 + 9 = 14
loss.backward()
# d(sum(x²))/dx = 2x = [2, 4, 6]
print(f"x = {x.tolist()}")
print(f"y = x² = {y.tolist()}")
print(f"d(sum(x²))/dx = {x.grad.tolist()}")  # [2.0, 4.0, 6.0]

# 實際場景中，loss function 的輸出都是純量，
# 所以通常不會遇到這個問題。


# ─────────────────────────────────────────────────────────────
# 2.5 停止梯度追蹤
# ─────────────────────────────────────────────────────────────
print("\n\n📌 2.5 停止梯度追蹤")
print("-" * 40)

# 有時候不需要計算梯度（例如：模型推論、評估時）
# 停止追蹤可以節省記憶體和加速計算

x = torch.tensor(5.0, requires_grad=True)

# 方法一：torch.no_grad() 上下文管理器（最常用）
with torch.no_grad():
    y = x * 2
    print(f"no_grad 內：y.requires_grad = {y.requires_grad}")  # False

# 方法二：detach() — 從計算圖中分離
y = x * 2
z = y.detach()  # z 不再追蹤梯度
print(f"detach 後：z.requires_grad = {z.requires_grad}")  # False

# 方法三：torch.inference_mode()（PyTorch 2.0+ 推薦）
with torch.inference_mode():
    y = x * 2
    print(f"inference_mode 內：y.requires_grad = {y.requires_grad}")

print("""
  使用時機：
  - 訓練時 → 需要梯度（requires_grad=True）
  - 驗證/測試時 → 不需要梯度（用 torch.no_grad()）
  - 正式推論時 → 不需要梯度（用 torch.inference_mode()）
""")


# ─────────────────────────────────────────────────────────────
# 2.6 實際案例：用 Autograd 實作線性迴歸
# ─────────────────────────────────────────────────────────────
print("\n📌 2.6 實際案例：用 Autograd 實作線性迴歸")
print("-" * 40)

# 跟第一章的手動版本相比，這次讓 PyTorch 自動計算梯度

# 產生模擬資料：y = 3x + 7 + noise
torch.manual_seed(42)
X = torch.rand(200, 1) * 10                    # 200 筆資料，範圍 0~10
y_true = 3.0 * X + 7.0 + torch.randn(200, 1)  # 加入隨機雜訊

# 初始化參數（這次用 requires_grad=True）
weight = torch.randn(1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)
learning_rate = 0.01

print(f"初始 weight = {weight.item():.4f}, bias = {bias.item():.4f}")
print(f"目標 weight = 3.0, bias = 7.0\n")

# 訓練迴圈
for epoch in range(100):
    # ─── 前向傳播 ───
    y_pred = weight * X + bias

    # ─── 計算損失 ───
    loss = ((y_pred - y_true) ** 2).mean()  # MSE

    # ─── 反向傳播（自動計算梯度）───
    loss.backward()
    # 此時 weight.grad 和 bias.grad 已經自動算好了！

    # ─── 更新參數 ───
    # 注意：更新時要在 no_grad() 裡面，因為更新操作不需要追蹤梯度
    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad

    # ─── 梯度清零（非常重要！）───
    weight.grad.zero_()
    bias.grad.zero_()

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} "
              f"| weight: {weight.item():.4f} | bias: {bias.item():.4f}")

print(f"\n最終結果：weight = {weight.item():.4f}（目標 3.0）"
      f", bias = {bias.item():.4f}（目標 7.0）")


# ─────────────────────────────────────────────────────────────
# 2.7 計算圖的生命週期
# ─────────────────────────────────────────────────────────────
print("\n\n📌 2.7 計算圖的生命週期")
print("-" * 40)

print("""
計算圖在每次 backward() 後會被釋放（預設行為）。
這代表每次訓練迭代都會重建計算圖。

  前向傳播 → 建立計算圖
  backward() → 計算梯度 + 釋放計算圖
  下一次前向傳播 → 重新建立計算圖
  ...

如果需要多次 backward()，要設定 retain_graph=True：
""")

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward(retain_graph=True)  # 保留計算圖
print(f"第一次 backward: x.grad = {x.grad}")  # 3 * 2² = 12

x.grad.zero_()
y.backward()  # 第二次 backward（計算圖會被釋放）
print(f"第二次 backward: x.grad = {x.grad}")  # 12

# y.backward()  # 第三次會報錯，因為計算圖已釋放


# ─────────────────────────────────────────────────────────────
# 2.8 常見錯誤與除錯
# ─────────────────────────────────────────────────────────────
print("\n\n📌 2.8 常見錯誤與除錯")
print("-" * 40)

print("""
錯誤 1：忘記清零梯度
────────────────────
  症狀：loss 不下降或震盪
  原因：梯度不斷累加
  解法：每次 backward 前呼叫 optimizer.zero_grad()

錯誤 2：在需要梯度的 Tensor 上做 in-place 操作
─────────────────────────────────────────────
  症狀：RuntimeError: a leaf Variable that requires grad ...
  原因：x += 1 這種 in-place 操作會破壞計算圖
  解法：用 x = x + 1 代替 x += 1

錯誤 3：Tensor 轉 NumPy 時忘記 detach
──────────────────────────────────────
  症狀：RuntimeError: Can't call numpy() on Tensor that requires grad
  解法：tensor.detach().cpu().numpy()

錯誤 4：混淆 .data 和 .detach()
──────────────────────────────
  .data 是不安全的（不會被 autograd 追蹤）
  .detach() 是安全的（推薦使用）
""")


# ─────────────────────────────────────────────────────────────
# 2.9 練習題
# ─────────────────────────────────────────────────────────────
print("\n📌 2.9 練習題")
print("-" * 40)
print("""
練習 1：計算 f(x) = sin(x) 在 x = pi/4 的梯度
        提示：d(sin(x))/dx = cos(x)，cos(pi/4) ≈ 0.707

練習 2：計算 f(x, y) = x²y + y³ 在 (x=2, y=3) 的偏微分
        提示：df/dx = 2xy, df/dy = x² + 3y²

練習 3：修改 2.6 的線性迴歸，改成二次方程式 y = ax² + bx + c
        目標：a=1, b=-2, c=3
""")

# 練習 1 參考答案
x = torch.tensor(torch.pi / 4, requires_grad=True)
y = torch.sin(x)
y.backward()
print(f"練習 1：d(sin(x))/dx at x=pi/4 = {x.grad.item():.4f}")
print(f"         cos(pi/4) = {torch.cos(torch.tensor(torch.pi/4)).item():.4f}")

print("\n\n" + "=" * 60)
print("第二章結束！下一章：用 nn.Module 建構神經網路")
print("=" * 60)

"""
=============================================================================
第九章：模型部署
=============================================================================

訓練好的模型只是第一步！
要讓模型發揮價值，必須部署到可以實際使用的環境中。

部署的方式：
──────────

  ┌─────────────────────────────────────────────────────┐
  │ 本地推論                                             │
  │ model.eval() + torch.no_grad()                      │
  │ 適用：快速測試、桌面應用                               │
  ├─────────────────────────────────────────────────────┤
  │ TorchScript (JIT)                                   │
  │ 把模型編譯成可序列化的格式                              │
  │ 適用：不需要 Python 的環境（C++/移動端）                │
  ├─────────────────────────────────────────────────────┤
  │ ONNX                                                │
  │ 通用模型格式，可在多種框架中執行                         │
  │ 適用：跨框架部署、TensorRT 加速                        │
  ├─────────────────────────────────────────────────────┤
  │ REST API                                            │
  │ 用 Flask/FastAPI 包成 HTTP 服務                      │
  │ 適用：後端伺服器、微服務架構                            │
  └─────────────────────────────────────────────────────┘

本章學習目標：
─────────────
✓ 模型的儲存與載入
✓ TorchScript 匯出
✓ ONNX 匯出
✓ 推論最佳化
✓ Flask/FastAPI 服務化
✓ 批次推論
=============================================================================
"""

import torch
import torch.nn as nn
import os
import time
import json

print("=" * 60)
print("第九章：模型部署")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立一個範例模型
class ImageClassifier(nn.Module):
    """範例模型：圖片分類器"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = ImageClassifier(num_classes=10).to(device)


# ─────────────────────────────────────────────────────────────
# 9.1 模型的儲存與載入
# ─────────────────────────────────────────────────────────────
print("\n📌 9.1 模型的儲存與載入")
print("-" * 40)

save_dir = './saved_models'
os.makedirs(save_dir, exist_ok=True)

# ── 方法一：只儲存參數（推薦！）──
# 優點：檔案小、載入時可以修改模型結構
# 缺點：載入時需要模型的類別定義
model_path = os.path.join(save_dir, 'model_weights.pth')
torch.save(model.state_dict(), model_path)
print(f"方法一：儲存模型參數 → {model_path}")
print(f"  檔案大小: {os.path.getsize(model_path) / 1024:.1f} KB")

# 載入參數
loaded_model = ImageClassifier(num_classes=10)  # 先建立模型結構
loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
loaded_model.eval()
print("  載入成功！")

# ── 方法二：儲存整個模型（不推薦）──
# 優點：不需要模型類別定義
# 缺點：檔案大、跟 Python 版本綁定、有安全風險
model_full_path = os.path.join(save_dir, 'model_full.pth')
torch.save(model, model_full_path)
print(f"\n方法二：儲存完整模型 → {model_full_path}")
print(f"  檔案大小: {os.path.getsize(model_full_path) / 1024:.1f} KB")

# ── 方法三：儲存 Checkpoint（訓練中斷後恢復）──
checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

checkpoint = {
    'epoch': 42,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.123,
    'accuracy': 95.5,
    'config': {
        'num_classes': 10,
        'learning_rate': 0.001,
    }
}
torch.save(checkpoint, checkpoint_path)
print(f"\n方法三：儲存 Checkpoint → {checkpoint_path}")

# 從 Checkpoint 恢復
ckpt = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_epoch = ckpt['epoch']
print(f"  恢復到 Epoch {start_epoch}，Loss={ckpt['loss']:.3f}")


# ─────────────────────────────────────────────────────────────
# 9.2 TorchScript（JIT 編譯）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 9.2 TorchScript（JIT 編譯）")
print("-" * 40)

print("""
  TorchScript 的作用：
  - 把 PyTorch 模型轉成獨立的格式
  - 可以在沒有 Python 的環境中執行（C++ runtime）
  - 適用於移動端（iOS/Android）和嵌入式設備
  - 可以做一些編譯優化，加速推論
""")

model.eval()

# 方法一：torch.jit.trace（追蹤法）
# 用一個範例輸入「跑一遍」模型，記錄所有操作
dummy_input = torch.rand(1, 3, 32, 32).to(device)
traced_model = torch.jit.trace(model, dummy_input)

script_path = os.path.join(save_dir, 'model_traced.pt')
traced_model.save(script_path)
print(f"Traced 模型 → {script_path}")
print(f"  檔案大小: {os.path.getsize(script_path) / 1024:.1f} KB")

# 載入並使用（不需要模型類別定義！）
loaded_traced = torch.jit.load(script_path)
output = loaded_traced(dummy_input)
print(f"  推論結果形狀: {output.shape}")

# 方法二：torch.jit.script（腳本法）
# 直接分析 Python 程式碼，轉成 TorchScript
# 支援 if/for 等控制流程（trace 不支援）
class ModelWithLogic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        out = self.linear(x)
        # trace 無法捕捉這種條件分支
        if out.sum() > 0:
            return out * 2
        else:
            return out

scripted_model = torch.jit.script(ModelWithLogic())
print(f"\nScripted 模型（支援控制流程）：{type(scripted_model)}")

print("""
  trace vs script 比較：
  ┌──────────┬─────────────────────────────┐
  │ trace    │ 簡單，但不支援動態控制流程     │
  │ script   │ 支援 if/for，但對程式碼有限制  │
  └──────────┴─────────────────────────────┘
  建議：先試 trace，不行再用 script
""")


# ─────────────────────────────────────────────────────────────
# 9.3 ONNX 匯出
# ─────────────────────────────────────────────────────────────
print("\n\n📌 9.3 ONNX 匯出")
print("-" * 40)

print("""
  ONNX（Open Neural Network Exchange）：
  - 通用的神經網路模型格式
  - 可以在多種推論引擎上執行：
    ・ONNX Runtime（微軟，跨平台）
    ・TensorRT（NVIDIA，GPU 加速）
    ・OpenVINO（Intel，CPU 優化）
    ・Core ML（Apple，iOS/Mac）
""")

model.eval()
dummy_input = torch.rand(1, 3, 32, 32).to(device)
onnx_path = os.path.join(save_dir, 'model.onnx')

try:
    torch.onnx.export(
        model,                          # 模型
        dummy_input,                    # 範例輸入
        onnx_path,                      # 輸出路徑
        export_params=True,             # 包含模型參數
        opset_version=13,               # ONNX 版本
        do_constant_folding=True,       # 常數摺疊優化
        input_names=['input'],          # 輸入節點名稱
        output_names=['output'],        # 輸出節點名稱
        dynamic_axes={                  # 動態批次大小
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"ONNX 匯出成功 → {onnx_path}")
    print(f"  檔案大小: {os.path.getsize(onnx_path) / 1024:.1f} KB")

    # 驗證 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX 模型驗證通過！")
    except ImportError:
        print("  （安裝 onnx 套件可以驗證模型：pip install onnx）")

    # 使用 ONNX Runtime 推論
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        np_input = dummy_input.cpu().numpy()
        result = session.run(None, {input_name: np_input})
        print(f"  ONNX Runtime 推論結果形狀: {result[0].shape}")
    except ImportError:
        print("  （安裝 onnxruntime 可以做推論：pip install onnxruntime）")

except Exception as e:
    print(f"  ONNX 匯出失敗: {e}")


# ─────────────────────────────────────────────────────────────
# 9.4 推論最佳化
# ─────────────────────────────────────────────────────────────
print("\n\n📌 9.4 推論最佳化")
print("-" * 40)

model.eval()
dummy_input = torch.rand(1, 3, 32, 32).to(device)

# 技巧一：torch.no_grad()（必做！）
print("技巧一：torch.no_grad()")
start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)
elapsed = time.time() - start
print(f"  100 次推論耗時: {elapsed:.4f} 秒")

# 技巧二：torch.inference_mode()（更快！PyTorch 2.0+）
print("\n技巧二：torch.inference_mode()")
start = time.time()
with torch.inference_mode():
    for _ in range(100):
        _ = model(dummy_input)
elapsed = time.time() - start
print(f"  100 次推論耗時: {elapsed:.4f} 秒")

# 技巧三：torch.compile()（PyTorch 2.0+ 的殺手級功能）
print("\n技巧三：torch.compile()（PyTorch 2.0+）")
try:
    compiled_model = torch.compile(model)
    # 第一次推論會比較慢（編譯中）
    with torch.inference_mode():
        _ = compiled_model(dummy_input)  # warmup

    start = time.time()
    with torch.inference_mode():
        for _ in range(100):
            _ = compiled_model(dummy_input)
    elapsed = time.time() - start
    print(f"  100 次推論耗時: {elapsed:.4f} 秒")
except Exception as e:
    print(f"  torch.compile 不可用: {e}")

# 技巧四：半精度推論（FP16）
print("\n技巧四：半精度推論（FP16）")
if device.type == 'cuda':
    model_fp16 = model.half()  # 轉成 FP16
    input_fp16 = dummy_input.half()
    start = time.time()
    with torch.inference_mode():
        for _ in range(100):
            _ = model_fp16(input_fp16)
    elapsed = time.time() - start
    print(f"  FP16 100 次推論耗時: {elapsed:.4f} 秒")
else:
    print("  FP16 加速主要在 GPU 上有效")

# 技巧五：批次推論（充分利用 GPU）
print("\n技巧五：批次推論（Batch Inference）")
batch_sizes = [1, 4, 16, 64]
for bs in batch_sizes:
    batch_input = torch.rand(bs, 3, 32, 32).to(device)
    start = time.time()
    with torch.inference_mode():
        for _ in range(100):
            _ = model(batch_input)
    elapsed = time.time() - start
    throughput = bs * 100 / elapsed
    print(f"  batch_size={bs:2d}: {elapsed:.4f}s, "
          f"吞吐量={throughput:.0f} images/sec")


# ─────────────────────────────────────────────────────────────
# 9.5 用 Flask 建立推論 API（程式碼範例）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 9.5 推論 API（Flask 範例程式碼）")
print("-" * 40)

# 以下是 Flask API 的完整程式碼，不在此執行
flask_code = '''
# === flask_api.py ===
# 執行：python flask_api.py
# 測試：curl -X POST -F "image=@test.jpg" http://localhost:5000/predict

from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# 載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier(num_classes=10)
model.load_state_dict(torch.load("saved_models/model_weights.pth",
                                  weights_only=True,
                                  map_location=device))
model.to(device)
model.eval()

# 定義前處理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    return jsonify({
        "class": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2),
        "probabilities": {
            cls: round(p, 4)
            for cls, p in zip(classes, probs[0].cpu().tolist())
        }
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
'''

print("Flask API 範例（程式碼已生成，可複製使用）：")
print("  - POST /predict → 上傳圖片，回傳分類結果")
print("  - GET /health → 健康檢查")
print("\n  完整程式碼請見上方字串，或建立 flask_api.py 檔案")

# 寫出 Flask API 檔案
flask_path = os.path.join(save_dir, 'flask_api_example.py')
with open(flask_path, 'w', encoding='utf-8') as f:
    f.write(flask_code)
print(f"  已寫出範例到 {flask_path}")


# ─────────────────────────────────────────────────────────────
# 9.6 用 FastAPI 建立推論 API（更現代的做法）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 9.6 推論 API（FastAPI 範例）")
print("-" * 40)

fastapi_code = '''
# === fastapi_server.py ===
# 執行：uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
# 文件：http://localhost:8000/docs  （自動生成 Swagger UI）

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI(title="Image Classifier API", version="1.0")

# 載入模型（啟動時執行一次）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None  # 在 startup 事件中載入

@app.on_event("startup")
async def load_model():
    global model
    model = ImageClassifier(num_classes=10)
    model.load_state_dict(torch.load("saved_models/model_weights.pth",
                                      weights_only=True,
                                      map_location=device))
    model.to(device)
    model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    return {
        "class": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}
'''

print("FastAPI 優勢：")
print("  - 自動生成 API 文件（Swagger UI）")
print("  - 非同步支援（async/await）")
print("  - 型別檢查")
print("  - 效能更好")


# ─────────────────────────────────────────────────────────────
# 9.7 部署清單
# ─────────────────────────────────────────────────────────────
print("\n\n📌 9.7 部署清單")
print("-" * 40)

print("""
  部署前的檢查清單：
  ┌────┬──────────────────────────────────────┐
  │ □  │ model.eval() 已設定                    │
  │ □  │ 使用 torch.no_grad() 或 inference_mode │
  │ □  │ 前處理與訓練時一致（Normalize 數值）     │
  │ □  │ 模型和輸入在同一個 device 上            │
  │ □  │ 批次推論已優化                          │
  │ □  │ 模型大小可接受（壓縮/量化）              │
  │ □  │ 錯誤處理已完善                          │
  │ □  │ API 有健康檢查端點                      │
  │ □  │ 日誌記錄已設定                          │
  └────┴──────────────────────────────────────┘
""")


# ─────────────────────────────────────────────────────────────
# 9.8 練習題
# ─────────────────────────────────────────────────────────────
print("\n📌 9.8 練習題")
print("-" * 40)
print("""
練習 1：儲存第五章的 CIFAR-10 CNN，然後用 TorchScript 匯出
        驗證匯出後的推論結果是否一致

練習 2：把模型匯出成 ONNX，用 ONNX Runtime 做推論
        比較 PyTorch 和 ONNX Runtime 的推論速度

練習 3：用 FastAPI 建立一個完整的圖片分類服務
        - 加入錯誤處理
        - 加入日誌記錄
        - 加入批次推論端點

練習 4：實作模型量化（Quantization）
        - torch.quantization.quantize_dynamic
        - 比較量化前後的模型大小和推論速度
""")

# 清理
import shutil
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
    print(f"\n已清理暫存目錄 {save_dir}")

print("\n" + "=" * 60)
print("第九章結束！下一章：最佳實踐與常見錯誤")
print("=" * 60)

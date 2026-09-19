"""
=============================================================================
第六章：實戰 — NLP 文字分類（RNN / LSTM）
=============================================================================

從圖片到文字：
─────────────
圖片 → 2D 空間結構 → CNN
文字 → 1D 序列結構 → RNN / LSTM

文字的處理流程：
──────────────
"我愛深度學習" → [我, 愛, 深度, 學習] → [42, 7, 256, 100] → [向量, 向量, 向量, 向量]
  原始文字          分詞(tokenize)         詞轉ID(encoding)       Embedding

RNN 的直覺理解：
──────────────
RNN 就像「有記憶的閱讀者」：
讀第一個字 → 記住
讀第二個字 → 結合之前的記憶，更新理解
讀第三個字 → 結合之前的記憶，更新理解
...
讀完最後一個字 → 基於累積的理解做出判斷

圖解：
  "這 部 電影 很 好看"
   │  │   │   │   │
   ▼  ▼   ▼   ▼   ▼
  [h0→h1→ h2→ h3→ h4] → 分類：正面評價
   RNN 的隱藏狀態不斷更新

LSTM vs 普通 RNN：
────────────────
普通 RNN 的問題：「長期記憶」很差（讀到後面會忘記前面的內容）
LSTM 的解決方案：加入「門控機制」，選擇性地記住/遺忘資訊

本章學習目標：
─────────────
✓ 文字前處理流程
✓ 詞嵌入（Word Embedding）
✓ RNN 和 LSTM 的使用
✓ 情感分析實戰
✓ 處理變長序列（padding + packing）
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

print("=" * 60)
print("第六章：實戰 — NLP 文字分類")
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
# 6.1 文字前處理
# ─────────────────────────────────────────────────────────────
print("\n📌 6.1 文字前處理")
print("-" * 40)

# 模擬電影評論資料集
reviews = [
    ("This movie is absolutely wonderful and amazing", 1),
    ("Terrible film waste of time and money", 0),
    ("Great acting and beautiful cinematography", 1),
    ("Boring movie nothing interesting happens", 0),
    ("I loved every minute of this masterpiece", 1),
    ("Awful terrible the worst movie ever made", 0),
    ("Fantastic story with brilliant performances", 1),
    ("Dull and predictable plot very disappointing", 0),
    ("An incredible journey that touches the heart", 1),
    ("So bad I walked out of the theater early", 0),
    ("Outstanding direction and superb screenplay", 1),
    ("Complete garbage not worth watching at all", 0),
    ("A beautiful and moving cinematic experience", 1),
    ("Horrible acting and a nonsensical story", 0),
    ("One of the best films I have ever seen", 1),
    ("Painfully slow and utterly boring movie", 0),
    ("Absolutely brilliant from start to finish", 1),
    ("A complete disaster avoid at all costs", 0),
    ("Heartwarming story with wonderful characters", 1),
    ("Worst film of the year by far", 0),
]

print(f"資料集大小: {len(reviews)} 筆評論")
print(f"正面: {sum(1 for _, l in reviews if l == 1)} 筆")
print(f"負面: {sum(1 for _, l in reviews if l == 0)} 筆")

# 步驟一：文字清洗 + 分詞
def tokenize(text):
    """簡單的英文分詞：轉小寫 + 用空格分割"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # 移除非字母字元
    return text.split()

sample = "This movie is ABSOLUTELY wonderful!!!"
tokens = tokenize(sample)
print(f"\n分詞範例：")
print(f"  原始：{sample}")
print(f"  分詞：{tokens}")

# 步驟二：建立詞彙表（Vocabulary）
all_tokens = []
for text, _ in reviews:
    all_tokens.extend(tokenize(text))

# 統計每個詞出現的次數
word_counts = Counter(all_tokens)
print(f"\n詞彙統計（前 10 常見詞）：")
for word, count in word_counts.most_common(10):
    print(f"  {word}: {count} 次")

# 建立 word → index 映射
# <PAD> = 0：填充用
# <UNK> = 1：未知詞
vocab = {'<PAD>': 0, '<UNK>': 1}
for word, count in word_counts.most_common():
    if count >= 1:  # 出現至少 1 次才加入詞彙表
        vocab[word] = len(vocab)

print(f"\n詞彙表大小: {len(vocab)}")
print(f"前 15 個詞: {dict(list(vocab.items())[:15])}")

# 步驟三：把文字轉成數字序列
def encode_text(text, vocab, max_length=20):
    """把文字轉成固定長度的 ID 序列"""
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    # Padding：不足長度補 0，超過長度截斷
    if len(ids) < max_length:
        ids = ids + [0] * (max_length - len(ids))
    else:
        ids = ids[:max_length]
    return ids

sample_encoded = encode_text("This movie is wonderful", vocab)
print(f"\n編碼範例：")
print(f"  文字：This movie is wonderful")
print(f"  ID：{sample_encoded}")


# ─────────────────────────────────────────────────────────────
# 6.2 建立 Dataset
# ─────────────────────────────────────────────────────────────
print("\n\n📌 6.2 建立 Dataset")
print("-" * 40)

class SentimentDataset(Dataset):
    def __init__(self, reviews, vocab, max_length=20):
        self.data = []
        for text, label in reviews:
            ids = encode_text(text, vocab, max_length)
            self.data.append((torch.tensor(ids, dtype=torch.long),
                            torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 為了教學目的，使用全部資料訓練（實際應用要分 train/test）
dataset = SentimentDataset(reviews, vocab)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 看一個 batch
sample_batch = next(iter(train_loader))
print(f"一個 batch 的文字 ID: {sample_batch[0].shape}")   # (4, 20)
print(f"一個 batch 的標籤: {sample_batch[1].tolist()}")


# ─────────────────────────────────────────────────────────────
# 6.3 詞嵌入（Word Embedding）
# ─────────────────────────────────────────────────────────────
print("\n\n📌 6.3 詞嵌入（Word Embedding）")
print("-" * 40)

print("""
為什麼需要 Embedding？

方法一：One-Hot Encoding（不好）
  "cat" → [0, 0, 1, 0, 0, ...]    # 1000 維的稀疏向量
  "dog" → [0, 0, 0, 1, 0, ...]
  問題：
  - 維度太高（詞彙量多大就多大）
  - 無法表達詞與詞的「相似度」
  - cat 和 dog 的距離 = cat 和 table 的距離

方法二：Word Embedding（好！）
  "cat" → [0.2, -0.5, 0.8, ...]   # 64 維的密集向量
  "dog" → [0.3, -0.4, 0.7, ...]
  優點：
  - 維度低（通常 64~300 維）
  - 語意相近的詞 → 向量也相近
  - 可以學到 king - man + woman ≈ queen
""")

# Embedding 層的使用
embedding = nn.Embedding(
    num_embeddings=len(vocab),   # 詞彙表大小
    embedding_dim=64,            # 每個詞用 64 維向量表示
    padding_idx=0                # PAD token 的向量固定為 0
)

# 把 word IDs 轉成向量
word_ids = torch.tensor([vocab.get('movie', 1), vocab.get('wonderful', 1)])
word_vectors = embedding(word_ids)
print(f"word IDs: {word_ids.tolist()}")
print(f"word vectors 形狀: {word_vectors.shape}")  # (2, 64)
print(f"word vector 範例: {word_vectors[0][:5].tolist()}")


# ─────────────────────────────────────────────────────────────
# 6.4 RNN 基礎
# ─────────────────────────────────────────────────────────────
print("\n\n📌 6.4 RNN 基礎")
print("-" * 40)

# nn.RNN 的輸入輸出：
# 輸入：(sequence_length, batch_size, input_size)  或 batch_first=True 時 (batch, seq, input)
# 輸出：
#   output: 每個時間步的隱藏狀態
#   h_n: 最後一個時間步的隱藏狀態

rnn = nn.RNN(
    input_size=64,        # 輸入維度（等於 embedding_dim）
    hidden_size=128,      # 隱藏狀態維度
    num_layers=1,         # RNN 層數
    batch_first=True      # 輸入格式：(batch, seq, features)
)

# 假設 batch=2, 序列長度=10, embedding_dim=64
fake_input = torch.rand(2, 10, 64)
output, h_n = rnn(fake_input)

print(f"RNN 輸入: {fake_input.shape}")      # (2, 10, 64)
print(f"RNN 輸出: {output.shape}")          # (2, 10, 128) — 每個時間步
print(f"最終隱藏: {h_n.shape}")             # (1, 2, 128) — 最後時間步


# ─────────────────────────────────────────────────────────────
# 6.5 LSTM — 長短期記憶網路
# ─────────────────────────────────────────────────────────────
print("\n\n📌 6.5 LSTM — 長短期記憶網路")
print("-" * 40)

print("""
LSTM 的門控機制（直覺理解）：

  ┌──────────────────────────────────────┐
  │ 遺忘門（Forget Gate）：決定要忘記多少   │
  │   「這段描述天氣，跟情感無關，忘掉」    │
  │                                      │
  │ 輸入門（Input Gate）：決定要記住多少     │
  │   「wonderful 這個字很重要，記住」      │
  │                                      │
  │ 輸出門（Output Gate）：決定要輸出多少   │
  │   「根據目前理解，輸出正面信號」        │
  └──────────────────────────────────────┘

LSTM 比 RNN 好在哪？
→ 能有效處理長序列（100+ 個字）
→ 較不會出現梯度消失問題
""")

lstm = nn.LSTM(
    input_size=64,
    hidden_size=128,
    num_layers=2,          # 2 層 LSTM
    batch_first=True,
    bidirectional=True,    # 雙向 LSTM（從前讀到後 + 從後讀到前）
    dropout=0.3            # 層間 dropout
)

fake_input = torch.rand(2, 10, 64)
output, (h_n, c_n) = lstm(fake_input)

print(f"LSTM 輸入: {fake_input.shape}")     # (2, 10, 64)
print(f"LSTM 輸出: {output.shape}")         # (2, 10, 256) — 雙向所以 128*2
print(f"最終隱藏 h_n: {h_n.shape}")         # (4, 2, 128) — 2層*2方向
print(f"最終細胞 c_n: {c_n.shape}")         # (4, 2, 128)


# ─────────────────────────────────────────────────────────────
# 6.6 完整模型：LSTM 情感分析器
# ─────────────────────────────────────────────────────────────
print("\n\n📌 6.6 完整模型：LSTM 情感分析器")
print("-" * 40)

class SentimentLSTM(nn.Module):
    """
    LSTM 情感分析模型

    處理流程：
    文字ID → Embedding → LSTM → 取最後隱藏狀態 → FC → Sigmoid → 正/負面

    架構圖：
    [word_ids] → [Embedding 64d] → [LSTM 128d] → [FC 64] → [FC 1] → 正/負
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 雙向 LSTM 的隱藏維度 = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)            # 二元分類
        )

    def forward(self, x):
        # x: (batch, seq_len) — word IDs

        # Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # LSTM: (batch, seq_len, embed_dim) → output, (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # 取最後一個時間步的輸出（雙向需要拼接）
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        # 取最後一層的前向和反向隱藏狀態
        forward_hidden = h_n[-2]    # (batch, hidden_dim)
        backward_hidden = h_n[-1]   # (batch, hidden_dim)
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        # hidden: (batch, hidden_dim * 2)

        # 分類
        output = self.classifier(hidden)  # (batch, 1)
        return output.squeeze(1)          # (batch,)


# 建立模型
model = SentimentLSTM(
    vocab_size=len(vocab),
    embed_dim=64,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3
).to(device)

print(f"模型結構：\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"\n總參數量: {total_params:,}")


# ─────────────────────────────────────────────────────────────
# 6.7 訓練情感分析模型
# ─────────────────────────────────────────────────────────────
print("\n\n📌 6.7 訓練情感分析模型")
print("-" * 40)

criterion = nn.BCEWithLogitsLoss()     # 二元分類
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100  # 資料量小，需要更多 epochs

print("訓練中：")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for text_ids, labels in train_loader:
        text_ids = text_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(text_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if epoch % 20 == 0:
        accuracy = 100.0 * correct / total
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%")


# ─────────────────────────────────────────────────────────────
# 6.8 測試模型
# ─────────────────────────────────────────────────────────────
print("\n\n📌 6.8 測試模型")
print("-" * 40)

def predict_sentiment(text, model, vocab, device):
    """預測一段文字的情感"""
    model.eval()
    ids = encode_text(text, vocab)
    tensor = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    sentiment = "正面" if prob > 0.5 else "負面"
    return sentiment, prob

# 測試一些新的評論
test_reviews = [
    "This is a wonderful and amazing movie",
    "Terrible awful waste of my time",
    "Great film loved the story",
    "Boring and dull nothing happens",
    "The best movie of the year",
]

print("情感預測結果：")
for review in test_reviews:
    sentiment, prob = predict_sentiment(review, model, vocab, device)
    print(f"  [{sentiment} {prob:.2f}] {review}")


# ─────────────────────────────────────────────────────────────
# 6.9 進階：處理變長序列
# ─────────────────────────────────────────────────────────────
print("\n\n📌 6.9 進階：處理變長序列（Packed Sequences）")
print("-" * 40)

print("""
問題：不同句子長度不同，但 batch 需要統一長度
     → 短句子後面填充了很多 <PAD>
     → LSTM 會白白處理這些無意義的 <PAD>

解決方案：Pack Padded Sequence
     → 告訴 LSTM 每個句子的實際長度
     → LSTM 只處理有效的部分，效率更高

流程：
  1. 記錄每個句子的實際長度
  2. 按長度排序（由長到短）
  3. pack_padded_sequence() 壓縮
  4. 餵給 LSTM
  5. pad_packed_sequence() 還原
""")

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 示範
sequences = torch.tensor([
    [1, 2, 3, 4, 5, 0, 0],    # 長度 5
    [6, 7, 8, 0, 0, 0, 0],    # 長度 3
    [9, 10, 11, 12, 0, 0, 0], # 長度 4
])
lengths = torch.tensor([5, 3, 4])

# 需要按長度排序
sorted_lengths, sorted_idx = lengths.sort(descending=True)
sorted_sequences = sequences[sorted_idx]
print(f"排序後序列:\n{sorted_sequences}")
print(f"排序後長度: {sorted_lengths.tolist()}")

# Embedding
embed = nn.Embedding(20, 8, padding_idx=0)
embedded = embed(sorted_sequences)   # (3, 7, 8)

# Pack
packed = pack_padded_sequence(embedded, sorted_lengths.cpu(), batch_first=True)
print(f"\nPacked data 形狀: {packed.data.shape}")

# 餵給 LSTM
lstm_small = nn.LSTM(8, 16, batch_first=True)
packed_output, (h_n, c_n) = lstm_small(packed)

# Unpack
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
print(f"Unpacked output 形狀: {output.shape}")


# ─────────────────────────────────────────────────────────────
# 6.10 練習題
# ─────────────────────────────────────────────────────────────
print("\n\n📌 6.10 練習題")
print("-" * 40)
print("""
練習 1：把 LSTM 換成 GRU（nn.GRU），比較訓練效果
        GRU 是 LSTM 的簡化版，參數更少

練習 2：加入 Attention 機制：
        - 不只看最後一個隱藏狀態，而是「注意」所有時間步
        - 計算每個時間步的重要性權重
        - 加權平均所有隱藏狀態
        提示：attn_weights = softmax(h @ query)
              context = sum(attn_weights * h)

練習 3：使用預訓練的 Word Embedding：
        - 下載 GloVe 或 Word2Vec
        - 用預訓練向量初始化 Embedding 層
        - 比較隨機初始化 vs 預訓練初始化的效果

練習 4：把資料集換成中文情感分析：
        - 使用 jieba 分詞
        - 重建中文詞彙表
        - 訓練中文情感分析模型
""")

print("\n" + "=" * 60)
print("第六章結束！下一章：遷移學習")
print("=" * 60)

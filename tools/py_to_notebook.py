"""
=============================================================================
把 chapters/*.py 教材轉換成 Jupyter Notebook (.ipynb)
=============================================================================

轉換規則：
- 模組最上方的 docstring  → Markdown 說明頁（標題、學習目標）
- `# ─────` 包住的小節標題 → Markdown 標題 (## N.M 標題)
- 其餘程式碼             → Code cell，過長的小節會依註解區塊再切開

因為小節標題已經變成 Markdown，預設會把純粹用來排版的標頭 print 拿掉：
- `print("\\n📌 1.1 建立 Tensor...")` 以及緊接在後面的 `print("-" * 40)`
- 章節開頭 banner 的 `print("=" * 60)` 與章名那行（版本資訊等其他 print 保留）
加上 --keep-headers 可以保留它們。

另外會自動補上 Google Colab 需要的東西：
- 開頭的 "Open in Colab" 徽章
- 「執行環境設定」cell：偵測 Colab、補裝缺少的套件、檢查 GPU（本機執行會略過安裝）
- notebook metadata 的 colab 設定，訓練章節加上 accelerator: GPU

切割用 ast 解析出「最外層敘述」的行範圍，所以不會切在字串、class、def 中間。

用法：
    python3 tools/py_to_notebook.py                 # 轉換全部章節
    python3 tools/py_to_notebook.py chapters/01_tensors.py
    python3 tools/py_to_notebook.py --keep-headers  # 保留標頭 print
=============================================================================
"""

import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER_DIR = REPO_ROOT / "chapters"
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

# 一條只由 ─ = - 組成的註解橫線，例如 "# ──────────────"
RULE_COMMENT = re.compile(r"^#\s*[─=\-—_]{5,}\s*$")
# 純橫線（docstring 內用）
RULE_LINE = re.compile(r"^\s*[─=\-—_]{5,}\s*$")
# 小節標題，例如 "1.1 建立 Tensor 的各種方式"
SECTION_TITLE = re.compile(r"^\d+\.\d+\s+\S")

# 一個 code cell 累積超過這個行數後，遇到新的註解區塊就切開
SPLIT_AFTER_LINES = 14

# ── Colab 設定 ───────────────────────────────────────────────
# 產生 "Open in Colab" 徽章用的 repo 位置
COLAB_REPO = "ChunPingWang/pytorach-tutorial"
COLAB_BRANCH = "main"

# 各章在 Colab 需要額外 pip install 的套件（torch / torchvision / numpy 已內建）
COLAB_EXTRAS = {
    "09_deployment": ["onnx", "onnxruntime"],
}

# 會實際訓練模型、建議開 GPU 執行階段的章節
GPU_CHAPTERS = {
    "05_cnn_image_classification",
    "06_nlp_text_classification",
    "07_transfer_learning",
    "08_gan",
}

# 需要提醒使用者「會下載資料」的章節
DOWNLOAD_NOTES = {
    "05_cnn_image_classification": "第一次執行會下載 CIFAR-10 資料集（約 170 MB）到 `./data`。",
    "07_transfer_learning": "第一次執行會下載 ResNet18 預訓練權重（約 45 MB）。",
    "08_gan": "第一次執行會下載 MNIST 資料集（約 10 MB）到 `./data`。",
    "09_deployment": "本章會在工作目錄產生 `.pth` / `.onnx` 等模型檔。",
}

# 小節標頭 print，例如 print("\n\n📌 1.2 資料型別（dtype）")
HEADER_PRINT = re.compile(r"""^print\(\s*f?["'](?:\\n)*\s*📌""")
# 排版用的橫線 print，例如 print("-" * 40)
RULER_PRINT = re.compile(r"""^print\(\s*["'][-=]["']\s*\*\s*\d+\s*\)\s*$""")
# 章節 banner 裡的章名，例如 print("第一章：PyTorch Tensor 張量基礎")
CHAPTER_TITLE_PRINT = re.compile(r"""^print\(\s*f?["']第.{1,3}章[：:]""")


# ─────────────────────────────────────────────────────────────
# docstring → Markdown
# ─────────────────────────────────────────────────────────────
def docstring_to_markdown(doc: str) -> str:
    """把章節開頭的 docstring 轉成 Markdown。

    - 文字下方那條 ──── 橫線 → 視為標題底線，升級成 ### 標題
    - 最上方的 ==== 橫線包住的那行 → 主標題 #
    - ✓ 開頭的行 → Markdown 清單
    - 縮排 4 格以上的 ASCII 圖 → 保持縮排（Markdown 會當成程式區塊）
    """
    lines = doc.strip("\n").split("\n")
    out: list[str] = []
    title_done = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # 純橫線：若下一行有文字且再下一行也是橫線 → 這是主標題的外框
        if RULE_LINE.match(line):
            if not title_done and i + 1 < len(lines) and lines[i + 1].strip():
                nxt = lines[i + 1].strip()
                if i + 2 < len(lines) and RULE_LINE.match(lines[i + 2]):
                    out.append(f"# {nxt}")
                    out.append("")
                    title_done = True
                    i += 3
                    continue
            i += 1
            continue

        # 文字後面接橫線 → 小標題
        if (
            line.strip()
            and i + 1 < len(lines)
            and RULE_LINE.match(lines[i + 1])
            and not line.startswith(" ")
        ):
            heading = line.strip().rstrip("：:")
            if not title_done:
                out.append(f"# {heading}")
                title_done = True
            else:
                out.append("")
                out.append(f"### {heading}")
            out.append("")
            i += 2
            continue

        # ✓ 開頭 → 清單
        if line.lstrip().startswith("✓"):
            out.append(f"- {line.strip()[1:].strip()}")
            i += 1
            continue

        out.append(line)
        i += 1

    # 壓掉連續空行
    cleaned: list[str] = []
    for line in out:
        if not line.strip() and cleaned and not cleaned[-1].strip():
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip("\n")


# ─────────────────────────────────────────────────────────────
# Colab 徽章與環境設定 cell
# ─────────────────────────────────────────────────────────────
def colab_badge(stem: str) -> str:
    url = (
        f"https://colab.research.google.com/github/{COLAB_REPO}"
        f"/blob/{COLAB_BRANCH}/notebooks/{stem}.ipynb"
    )
    return (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"
    )


def colab_cells(stem: str) -> list[dict]:
    """產生「執行環境設定」的 Markdown + Code cell。

    設定 cell 在 Colab 會自動補裝缺少的套件、提醒開 GPU；
    在本機執行則只印出環境資訊，不會動到你的套件。
    """
    extras = COLAB_EXTRAS.get(stem, [])

    notes = ["> 這個 Notebook 在 **Google Colab** 和**本機 Jupyter** 都能直接執行。"]
    if stem in GPU_CHAPTERS:
        notes.append(
            "> ⚡ 本章會實際訓練模型，在 Colab 請先開 GPU："
            "**執行階段 → 變更執行階段類型 → T4 GPU**。"
        )
    if extras:
        notes.append(f"> 📦 需要額外套件：`{'`、`'.join(extras)}`，下方 cell 會自動安裝。")
    if stem in DOWNLOAD_NOTES:
        notes.append(f"> 💾 {DOWNLOAD_NOTES[stem]}")
    notes.append("> Colab 的檔案在執行階段結束後會清空，需要保留請下載或掛載 Google Drive。")

    # 每則說明中間夾一行 ">"，Markdown 才會分段而不是黏成一坨
    md = "## 🚀 執行環境設定\n\n" + "\n>\n".join(notes)

    if extras:
        required = ", ".join(f'"{p}"' for p in extras)
        install = f'''
REQUIRED = [{required}]  # 這章需要、但 Colab 沒有內建的套件
missing = [p for p in REQUIRED if importlib.util.find_spec(p) is None]

if missing:
    if IN_COLAB:
        print(f"安裝缺少的套件：{{', '.join(missing)}}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *missing], check=True
        )
    else:
        print(f"⚠️ 本機缺少套件：pip install {{' '.join(missing)}}")
'''
        imports = "import importlib.util\nimport subprocess\nimport sys\n"
    else:
        install = ""
        imports = "import importlib.util\n"

    code = f'''# 在 Colab 會自動補裝套件並檢查 GPU；在本機執行只會印出環境資訊
{imports}
IN_COLAB = importlib.util.find_spec("google.colab") is not None
{install}
import torch

print(f"執行環境：{{'Google Colab' if IN_COLAB else '本機'}}")
print(f"PyTorch 版本：{{torch.__version__}}")
print(f"CUDA 可用：{{torch.cuda.is_available()}}")

if torch.cuda.is_available():
    print(f"GPU：{{torch.cuda.get_device_name(0)}}")
elif IN_COLAB:
    print("⚠️ 目前是 CPU 執行階段，需要 GPU 請切換：執行階段 → 變更執行階段類型 → T4 GPU")
'''

    md_cell = make_cell("markdown", md)
    code_cell = make_cell("code", code.strip("\n"))
    for cell in (md_cell, code_cell):
        cell["metadata"]["colab_setup"] = True  # 驗證時要跳過這兩格
    return [md_cell, code_cell]


# ─────────────────────────────────────────────────────────────
# 找出純排版用的標頭 print
# ─────────────────────────────────────────────────────────────
def find_header_prints(src_lines: list[str]) -> set[int]:
    """回傳要拿掉的行號（1-based）。

    小節標題已經是 Markdown 了，這些 print 只會變成重複的雜訊：
    1. `print("\\n📌 ...")` 和緊接在後的橫線 print
    2. 章節開頭 banner 的上下兩條 `print("=" * 60)` 與章名那行
       （中間的 PyTorch 版本、CUDA 是否可用等資訊保留）
    """
    drop: set[int] = set()

    for i, raw in enumerate(src_lines):
        line = raw.strip()
        if not HEADER_PRINT.match(line):
            continue
        drop.add(i + 1)
        if i + 1 < len(src_lines) and RULER_PRINT.match(src_lines[i + 1].strip()):
            drop.add(i + 2)

    # 章節開頭 banner：橫線 → 章名 → ...（其他資訊）... → 橫線
    for i, raw in enumerate(src_lines):
        if not RULER_PRINT.match(raw.strip()):
            continue
        if i + 1 >= len(src_lines) or not CHAPTER_TITLE_PRINT.match(src_lines[i + 1].strip()):
            continue
        for j in range(i + 2, min(i + 8, len(src_lines))):
            if RULER_PRINT.match(src_lines[j].strip()):
                drop.update({i + 1, i + 2, j + 1})
                break
        break  # 只處理檔案最上方那一組

    return drop


# ─────────────────────────────────────────────────────────────
# 把原始碼切成 item（註解區塊 / 最外層敘述）
# ─────────────────────────────────────────────────────────────
class Item:
    """原始檔裡的一段內容：註解區塊或最外層敘述，記錄它的起訖行號。"""

    def __init__(self, kind: str, start: int, end: int, node=None):
        self.kind = kind      # 'comment' | 'code'
        self.start = start    # 1-based，含
        self.end = end        # 1-based，含
        self.node = node

    def lines(self, src_lines: list[str]) -> list[str]:
        return src_lines[self.start - 1 : self.end]


def build_items(src_lines: list[str], start_line: int, tree: ast.Module) -> list[Item]:
    """把原始碼切成 Item 串列。

    start_line 是 1-based，從這行開始掃（跳過 docstring）。
    用 ast 的 lineno/end_lineno 確保不會切進字串或 class/def 內部。
    """
    # 最外層敘述的起訖行（1-based，含頭含尾）
    starts: dict[int, tuple[int, ast.stmt]] = {}
    for node in tree.body:
        begin = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        starts[begin] = (node.end_lineno, node)

    items: list[Item] = []
    line_no = start_line
    total = len(src_lines)

    while line_no <= total:
        if line_no in starts:
            end, node = starts[line_no]
            items.append(Item("code", line_no, end, node))
            line_no = end + 1
            continue

        text = src_lines[line_no - 1]
        if not text.strip():
            line_no += 1
            continue

        if text.lstrip().startswith("#"):
            begin = line_no
            while (
                line_no <= total
                and src_lines[line_no - 1].strip()
                and src_lines[line_no - 1].lstrip().startswith("#")
                and line_no not in starts
            ):
                line_no += 1
            items.append(Item("comment", begin, line_no - 1))
            continue

        # 理論上不會走到這（ast 已涵蓋所有敘述），保險起見原樣保留
        items.append(Item("code", line_no, line_no))
        line_no += 1

    return items


def section_heading(block: list[str]) -> str | None:
    """判斷註解區塊是不是 `# ────` 包起來的小節標題，是的話回傳標題文字。"""
    if len(block) < 3:
        return None
    if not RULE_COMMENT.match(block[0].strip()):
        return None
    if not RULE_COMMENT.match(block[-1].strip()):
        return None
    body = [ln.strip().lstrip("#").strip() for ln in block[1:-1]]
    body = [ln for ln in body if ln]
    if not body:
        return None
    return "\n".join(body)


# ─────────────────────────────────────────────────────────────
# item → notebook cells
# ─────────────────────────────────────────────────────────────
def make_cell(cell_type: str, text: str) -> dict:
    lines = text.split("\n")
    source = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    cell = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def items_to_cells(items: list[Item], src_lines: list[str], drop: set[int]) -> list[dict]:
    """把 Item 串成 cells。

    一個 code cell 就是原始檔的一段連續行（buffer 只記起訖行號），
    所以原本的空行、縮排、註解位置都原封不動保留下來；
    只有 drop 裡的標頭 print 會被濾掉。
    """
    cells: list[dict] = []
    buffer: list[Item] = []   # 目前累積的 item
    pending: Item | None = None  # 還沒接到程式碼的註解區塊

    def flush():
        nonlocal buffer
        if buffer:
            kept = [
                src_lines[n - 1]
                for n in range(buffer[0].start, buffer[-1].end + 1)
                if n not in drop
            ]
            text = "\n".join(kept).strip("\n")
            if text:
                cells.append(make_cell("code", text))
        buffer = []

    def keep_pending():
        """把還沒配對的註解放進目前的 cell，避免內容遺失。"""
        nonlocal pending
        if pending is not None:
            buffer.append(pending)
            pending = None

    for item in items:
        if item.kind == "comment":
            heading = section_heading(item.lines(src_lines))
            if heading is not None:
                keep_pending()
                flush()
                title, *rest = heading.split("\n")
                md = f"## {title}"
                if rest:
                    md += "\n\n" + "\n".join(rest)
                cells.append(make_cell("markdown", md))
                continue

            # 一般註解：先暫存，等它後面的程式碼一起放進同一個 cell
            keep_pending()
            pending = item
            continue

        # code：決定要不要在這裡切開
        is_def = isinstance(item.node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        head = pending or item
        span = len(
            [ln for ln in src_lines[buffer[0].start - 1 : head.start - 1] if ln.strip()]
        ) if buffer else 0
        if buffer and (is_def or (pending is not None and span >= SPLIT_AFTER_LINES)):
            flush()

        keep_pending()
        buffer.append(item)

        # class / def 自成一個 cell，讀起來比較清楚
        if is_def:
            flush()

    keep_pending()
    flush()
    return cells


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def convert(path: Path, strip_headers: bool = True) -> tuple[dict, set[int]]:
    src = path.read_text(encoding="utf-8")
    src_lines = src.split("\n")
    tree = ast.parse(src)
    drop = find_header_prints(src_lines) if strip_headers else set()

    cells: list[dict] = []
    start_line = 1

    # 模組 docstring → Markdown 說明頁
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        doc_node = tree.body[0]
        # 徽章放在最上面，從 GitHub 開這個檔案時可以直接點去 Colab
        title_md = colab_badge(path.stem) + "\n\n" + docstring_to_markdown(doc_node.value.value)
        cells.append(make_cell("markdown", title_md))
        start_line = doc_node.end_lineno + 1
        tree.body = tree.body[1:]

    cells.extend(colab_cells(path.stem))

    items = build_items(src_lines, start_line, tree)
    cells.extend(items_to_cells(items, src_lines, drop))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "codemirror_mode": {"name": "ipython", "version": 3},
            },
            # Colab 專用設定：保留目錄側欄，訓練章節預設用 GPU 執行階段開啟
            "colab": {
                "name": f"{path.stem}.ipynb",
                "provenance": [],
                "toc_visible": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    if path.stem in GPU_CHAPTERS:
        notebook["metadata"]["accelerator"] = "GPU"
    return notebook, drop


def verify(path: Path, notebook: dict, drop: set[int]) -> None:
    """檢查轉換沒有遺失或改動任何一行程式碼。

    只有兩種東西可以消失：變成 Markdown 標題的 `# ────` 註解區塊，
    以及 drop 裡那些標頭 print。其餘每一行都必須原樣出現在 code cell 中。
    """
    src = path.read_text(encoding="utf-8")
    src_lines = src.split("\n")
    tree = ast.parse(src)
    start_line = 1
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        start_line = tree.body[0].end_lineno + 1
        tree.body = tree.body[1:]

    expected = []
    for item in build_items(src_lines, start_line, tree):
        lines = item.lines(src_lines)
        if item.kind == "comment" and section_heading(lines) is not None:
            continue  # 這塊會變成 Markdown 標題
        expected.extend(
            src_lines[n - 1].rstrip()
            for n in range(item.start, item.end + 1)
            if src_lines[n - 1].strip() and n not in drop
        )
    got: list[str] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if cell["metadata"].get("colab_setup"):
            continue  # 這是外加的環境設定 cell，不在原始檔裡
        for line in "".join(cell["source"]).split("\n"):
            if line.strip():
                got.append(line.rstrip())

    if expected != got:
        for i, (a, b) in enumerate(zip(expected, got)):
            if a != b:
                raise SystemExit(
                    f"{path.name}: 第 {i} 行不一致\n  原始: {a!r}\n  轉換: {b!r}"
                )
        raise SystemExit(
            f"{path.name}: 行數不一致 原始 {len(expected)} vs 轉換 {len(got)}"
        )


def main(argv: list[str]) -> None:
    strip_headers = "--keep-headers" not in argv
    args = [a for a in argv if not a.startswith("--")]
    targets = [Path(a) for a in args] or sorted(CHAPTER_DIR.glob("*.py"))
    NOTEBOOK_DIR.mkdir(exist_ok=True)

    for path in targets:
        notebook, drop = convert(path, strip_headers)
        verify(path, notebook, drop)
        out = NOTEBOOK_DIR / (path.stem + ".ipynb")
        out.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
        md = len(notebook["cells"]) - code
        removed = f"，移除 {len(drop)} 行標頭 print" if drop else ""
        print(f"✓ {out.relative_to(REPO_ROOT)}  ({md} markdown + {code} code cells{removed})")


if __name__ == "__main__":
    main(sys.argv[1:])

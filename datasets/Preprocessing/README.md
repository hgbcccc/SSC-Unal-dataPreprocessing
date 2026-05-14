# Dataset Preprocessing

将两个原始数据集（Spruce Bark Beetle / TreeAI）转换为 COCO 格式并按 **主动学习 (Active Learning)** 场景划分训练/验证/测试/未标注子集。

> Convert two raw datasets (Spruce Bark Beetle / TreeAI) into COCO format and split them into train/val/test/unlabeled subsets for **Active Learning** scenarios.

---

## 目录 / Contents

- [环境依赖 / Requirements](#环境依赖--requirements)
- [数据集结构 / Dataset Structures](#数据集结构--dataset-structures)
- [快速开始 / Quick Start](#快速开始--quick-start)
  - [1. Spruce-Bark-Beetle v1.0](#1-spruce-bark-beetle-v10)
  - [2. TreeAI v1.2](#2-treeai-v12)
- [输出结构 / Output Structure](#输出结构--output-structure)
- [可视化 / Visualization](#可视化--visualization)
- [数据分析 / Data Analysis](#数据分析--data-analysis)
- [配置说明 / Configuration](#配置说明--configuration)

---

## 环境依赖 / Requirements

```bash
pip install pillow matplotlib numpy
```

Python 标准库即可运行核心脚本，`PIL` 用于读取图像尺寸（TreeAI），`matplotlib`/`numpy` 仅用于分析绘图。

> Core scripts run on Python stdlib. `PIL` is needed for image-size reading (TreeAI), `matplotlib`/`numpy` only for analysis plots.

---

## 数据集结构 / Dataset Structures

### Spruce Bark Beetle（原始 / Raw）

```
datasets/Data_Set_Spruce_Bark_Beetle/
├── oblique/
│   ├── <Location>_<YYYYMMDD>/
│   │   ├── Images/          # *.jpg
│   │   └── Annotations/     # *.xml (VOC format)
│   │   ...
│   └── ...
├── vertical/
│   └── ...                  # 同上 / same structure
└── damage/                  # 已损坏，会被自动跳过 / damaged, auto-skipped
```

- 标注格式 / Annotation format: **Pascal VOC XML**
- 类别 / Classes: `Spruce`, `Pine`, `Birch`, `Aspen`, `Other`（从 `<tree>` 或 `<name>` 标签读取）

### TreeAI（原始 / Raw）

```
datasets/12_RGB_FullyLabeled_640/coco/
├── train/
│   ├── images/              # *.png
│   └── labels/              # *.txt (YOLO format)
├── val/
│   ├── images/
│   └── labels/
├── classes.txt              # label index → class_id 映射
└── class12_RGB_all_L.xlsx   # class_id → labels → class_name
```

- 标注格式 / Annotation format: **YOLO txt** (normalized cx cy w h)
- 类别 / Classes: 12 类（默认取 label 数量最多的 Top-10）

---

## 快速开始 / Quick Start

### 1. Spruce-Bark-Beetle v1.0

```bash
cd datasets/Preprocessing
python preprocess_sbb.py
```

**执行流程：**
1. 扫描 `oblique/` 和 `vertical/` 下所有子文件夹，自动跳过 `damage/`
2. 解析 VOC XML 标注，提取边界框与类别
3. 构建完整 COCO JSON（含地点、日期、视角等元信息）
4. 按配置比例随机划分 → `init_train / val / test / unlabeled`
5. 余数图片自动合并到 `unlabeled`
6. 硬链接（或复制）图片到对应子集目录，写入各子集的 COCO JSON

### 2. TreeAI v1.2

```bash
cd datasets/Preprocessing
python preprocess_treeai.py
```

**执行流程：**
1. 读取 `classes.txt` 和 `class12_RGB_all_L.xlsx` 获取类别映射
2. 按标注数量取 Top-K 类别（默认 K=10）
3. 解析 YOLO txt 标注，转换为 COCO bbox（`[x, y, width, height]`，像素坐标）
4. 过滤掉不含 Top-K 类别框的图片
5. 按配置比例随机划分 → `init_train / val / test / unlabeled`
6. 余数图片自动合并到 `unlabeled`
7. 硬链接（或复制）图片并写入 COCO JSON

---

## 输出结构 / Output Structure

运行后会在 `datasets/Preprocessing/` 下生成：

```
Spruce-Bark-Beetle v1.0/          # 或 TreeAI v1.2/
├── annotations/
│   ├── init_train.json           # COCO 格式，初始标注训练集
│   ├── val.json                   # 验证集
│   ├── test.json                  # 测试集
│   └── unlabeled.json             # 未标注池（仍有 GT，用于主动学习模拟）
├── init_train/                    # 图片文件（硬链接/复制）
│   ├── BKS_20220801_V_000001.jpg
│   └── ...
├── val/
├── test/
└── unlabeled/
```

**子集含义 / Split Semantics：**

| 子集 | 默认比例 | 用途 |
|------|---------|------|
| `init_train` | 5% | 初始标注训练集 / Initial labeled training set |
| `val` | 10% | 验证集 / Validation set |
| `test` | 10% | 测试集 / Test set |
| `unlabeled` | 75% | 未标注池（保留 GT 用于主动学习模拟）/ Unlabeled pool with GT held for AL simulation |

> `VAL_EQUALS_TEST = True` 时，test 集与 val 集完全相同。

---

## 可视化 / Visualization

### Spruce-Bark-Beetle

```bash
python visualize_coco_sbb.py \
  --images-dir "Spruce-Bark-Beetle v1.0/val" \
  --ann "Spruce-Bark-Beetle v1.0/annotations/val.json" \
  -n 20 \
  --out vis_sbb_val
```

### TreeAI

```bash
python visualize_coco_treeai.py \
  --images-dir "TreeAI v1.2/val" \
  --ann "TreeAI v1.2/annotations/val.json" \
  -n 20 \
  --out vis_treeai_val
```

**参数说明 / Arguments：**

| 参数 | 说明 |
|------|------|
| `--images-dir` | 图片所在目录（split 根目录，不含 `images/` 子目录） |
| `--ann` | 对应的 COCO JSON 标注文件 |
| `-n` | 随机抽取图片数量，默认 20 |
| `--seed` | 随机种子 |
| `--out` | 输出目录 |

---

## 数据分析 / Data Analysis

```bash
python analyze_sbb.py
```

生成以下内容：
- 总体统计（图片数、标注数、类别分布）
- 各子集分别统计
- 按地点（Location）的类别分布分析
- 4 张科学图表（保存在 `datasets/Preprocessing/figures/`）：
  - 子集图片数量柱状图
  - 类别分布饼图
  - 地点分布柱状图
  - 地点 × 类别堆叠柱状图

---

## 配置说明 / Configuration

编辑脚本顶部的 `Split Config` 区域即可调整参数：

```python
SEED = 20260504               # 随机种子 / Random seed
SPLIT_RATIOS = {
    "init_train": 0.05,       # 初始训练集比例
    "val": 0.10,               # 验证集比例
    "test": 0.10,              # 测试集比例
    "unlabeled": 0.75,         # 未标注池比例
}
VAL_EQUALS_TEST = False        # True 时 test == val
OUTPUT_DIRNAME = "Spruce-Bark-Beetle v1.0"   # 输出目录名

# TreeAI 额外参数 / TreeAI extra:
TOPK = 10                      # 保留标注数量最多的前 K 个类别
```

> **注意：** 四个比例之和应为 1.0。由于整除产生的余数图片会自动合并到 `unlabeled`，不会丢失数据。
>
> **Note:** Ratios should sum to 1.0. Remainder images from integer truncation are automatically merged into `unlabeled` — no data is lost.

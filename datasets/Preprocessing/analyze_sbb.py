# import json
# import os
# from collections import defaultdict

# # ===================== 配置 =====================
# ANN_DIR = "datasets/Preprocessing/Spruce-Bark-Beetle v1.0/annotations"
# SPLITS = ["init_train", "val", "test", "unlabeled"]
# # ==================================================

# def load_json(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def count_stats(coco):
#     img_ids = {im["id"] for im in coco["images"]}
#     num_images = len(img_ids)
#     num_anns = len(coco["annotations"])

#     cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
#     cat_counts = defaultdict(int)
#     for ann in coco["annotations"]:
#         name = cat_id_to_name[ann["category_id"]]
#         cat_counts[name] += 1

#     return num_images, num_anns, cat_counts

# def print_stats(title, num_imgs, num_anns, cat_counts):
#     print(f"\n===== {title} =====")
#     print(f"图片数量: {num_imgs}")
#     print(f"标注总数: {num_anns}")
#     if num_anns == 0:
#         print("无标注")
#         return

#     print("\n类别分布:")
#     for cls, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
#         ratio = cnt / num_anns * 100
#         print(f"  {cls:<12} {cnt:>4}  ({ratio:.1f}%)")

# def analyze_by_location(full_coco):
#     loc_images = defaultdict(list)
#     loc_anns = defaultdict(list)

#     img_id_to_loc = {im["id"]: im["loc"] for im in full_coco["images"]}
#     for im in full_coco["images"]:
#         loc_images[im["loc"]].append(im)
#     for ann in full_coco["annotations"]:
#         loc = img_id_to_loc[ann["image_id"]]
#         loc_anns[loc].append(ann)

#     cat_names = {cat["id"]: cat["name"] for cat in full_coco["categories"]}
#     print("\n\n==============================================")
#     print("                按地点分析")
#     print("==============================================")

#     for loc in sorted(loc_images.keys()):
#         imgs = loc_images[loc]
#         anns = loc_anns[loc]
#         num_imgs = len(imgs)
#         num_anns = len(anns)

#         cat_counts = defaultdict(int)
#         for ann in anns:
#             name = cat_names[ann["category_id"]]
#             cat_counts[name] += 1

#         print_stats(f"地点 {loc}", num_imgs, num_anns, cat_counts)

# def main():
#     # 加载所有标注
#     split_data = {}
#     for s in SPLITS:
#         p = os.path.join(ANN_DIR, f"{s}.json")
#         if os.path.exists(p):
#             split_data[s] = load_json(p)
#         else:
#             print(f"跳过: {p}")

#     # 构建全集
#     all_images = []
#     all_annotations = []
#     all_categories = None
#     for s in SPLITS:
#         if s not in split_data:
#             continue
#         c = split_data[s]
#         all_images.extend(c["images"])
#         all_annotations.extend(c["annotations"])
#         if all_categories is None:
#             all_categories = c["categories"]

#     full_coco = {
#         "images": all_images,
#         "annotations": all_annotations,
#         "categories": all_categories
#     }

#     # ===================== 总体统计 =====================
#     num_imgs, num_anns, cat_counts = count_stats(full_coco)
#     print("==============================================")
#     print("                总体统计")
#     print("==============================================")
#     print_stats("全部数据", num_imgs, num_anns, cat_counts)

#     # ===================== 按子集统计 =====================
#     print("\n\n==============================================")
#     print("                按子集统计")
#     print("==============================================")
#     for s in SPLITS:
#         if s not in split_data:
#             continue
#         imgs, anns, cats = count_stats(split_data[s])
#         print_stats(f"子集 {s}", imgs, anns, cats)

#     # ===================== 按地点分析 =====================
#     analyze_by_location(full_coco)

#     print("\n分析完成！")

# if __name__ == "__main__":
#     main()



import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ===================== 配置 =====================
ANN_DIR = "datasets/Preprocessing/Spruce-Bark-Beetle v1.0/annotations"
SPLITS = ["init_train", "val", "test", "unlabeled"]
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590', '#F3722C', '#F9C74F']
plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "savefig.dpi": 300
})
# ==================================================

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def count_stats(coco):
    img_ids = {im["id"] for im in coco["images"]}
    num_images = len(img_ids)
    num_anns = len(coco["annotations"])
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    cat_counts = defaultdict(int)
    for ann in coco["annotations"]:
        name = cat_id_to_name[ann["category_id"]]
        cat_counts[name] += 1
    return num_images, num_anns, cat_counts

def print_stats(title, num_imgs, num_anns, cat_counts):
    print(f"\n===== {title} =====")
    print(f"Image count: {num_imgs}")
    print(f"Total annotations: {num_anns}")
    if num_anns == 0:
        print("No annotations")
        return
    print("\nCategory distribution:")
    for cls, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        ratio = cnt / num_anns * 100
        print(f"  {cls:<12} {cnt:>4}  ({ratio:.1f}%)")

def analyze_by_location(full_coco):
    loc_images = defaultdict(list)
    loc_anns = defaultdict(list)
    img_id_to_loc = {im["id"]: im["loc"] for im in full_coco["images"]}
    for im in full_coco["images"]:
        loc_images[im["loc"]].append(im)
    for ann in full_coco["annotations"]:
        loc = img_id_to_loc[ann["image_id"]]
        loc_anns[loc].append(ann)
    cat_names = {cat["id"]: cat["name"] for cat in full_coco["categories"]}
    print("\n\n==============================================")
    print("                Analysis by Location")
    print("==============================================")
    for loc in sorted(loc_images.keys()):
        imgs = loc_images[loc]
        anns = loc_anns[loc]
        num_imgs = len(imgs)
        num_anns = len(anns)
        cat_counts = defaultdict(int)
        for ann in anns:
            name = cat_names[ann["category_id"]]
            cat_counts[name] += 1
        print_stats(f"Location {loc}", num_imgs, num_anns, cat_counts)

# ===================== 绘图：地点 × 类别堆叠柱状图 =====================
def plot_location_category_stacked(full_coco):
    save_dir = "datasets/Preprocessing/figures"
    os.makedirs(save_dir, exist_ok=True)

    locs = sorted({im["loc"] for im in full_coco["images"]})
    cat_names = sorted({cat["name"] for cat in full_coco["categories"]})

    loc_cat_counts = defaultdict(lambda: defaultdict(int))
    img_id_to_loc = {im["id"]: im["loc"] for im in full_coco["images"]}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in full_coco["categories"]}

    for ann in full_coco["annotations"]:
        loc = img_id_to_loc[ann["image_id"]]
        cat = cat_id_to_name[ann["category_id"]]
        loc_cat_counts[loc][cat] += 1

    data = {loc: [loc_cat_counts[loc].get(c, 0) for c in cat_names] for loc in locs}
    data_np = np.array([data[loc] for loc in locs]).T
    bottom = np.zeros(len(locs))
    width = 0.6

    plt.figure(figsize=(8, 5))
    for i, cat in enumerate(cat_names):
        values = data_np[i]
        plt.bar(locs, values, width, label=cat, bottom=bottom, color=COLORS[i % len(COLORS)])
        bottom += values

    plt.title("Annotation Count by Location and Category", fontweight="bold", fontsize=14)
    plt.ylabel("Total Annotations")
    plt.xlabel("Location")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "location_category_stacked.png"), bbox_inches='tight')
    plt.close()

# ===================== 全部绘图 =====================
def plot_scientific_stats(full_coco, split_data):
    save_dir = "datasets/Preprocessing/figures"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 子集图片数量
    split_names = [s for s in SPLITS if s in split_data]
    split_img_counts = [len(split_data[s]["images"]) for s in split_names]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(split_names, split_img_counts, color=COLORS[0], alpha=0.8)
    plt.title("Image Count per Dataset Split", fontweight='bold', fontsize=14)
    plt.ylabel("Number of Images")
    plt.grid(axis='y', linestyle='--')
    for b, v in zip(bars, split_img_counts):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+10, str(v), ha='center', fontweight='bold')
    plt.savefig(os.path.join(save_dir, "dataset_split_images.png"), bbox_inches='tight')
    plt.close()

    # 2. 类别分布
    _, total_anns, cat_counts = count_stats(full_coco)
    if total_anns > 0:
        labels = list(cat_counts.keys())
        sizes = list(cat_counts.values())
        plt.figure(figsize=(7, 7))
        plt.pie(sizes, labels=labels, colors=COLORS[:len(labels)], autopct='%1.1f%%', startangle=90)
        plt.title("Overall Category Distribution", fontweight='bold', fontsize=14)
        plt.savefig(os.path.join(save_dir, "category_distribution.png"), bbox_inches='tight')
        plt.close()

    # 3. 地点图片数量
    locs = sorted({im["loc"] for im in full_coco["images"]})
    loc_counts = [len([im for im in full_coco["images"] if im["loc"] == l]) for l in locs]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(locs, loc_counts, color=COLORS[1], alpha=0.85)
    plt.title("Image Count by Location", fontweight='bold', fontsize=14)
    plt.ylabel("Number of Images")
    plt.grid(axis='y', linestyle='--')
    for b, v in zip(bars, loc_counts):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+10, str(v), ha='center', fontweight='bold')
    plt.savefig(os.path.join(save_dir, "location_distribution.png"), bbox_inches='tight')
    plt.close()

    # 4. 新增：地点 × 类别堆叠柱状图
    plot_location_category_stacked(full_coco)

    print(f"\n✅ All 4 scientific figures saved to: {save_dir}")

# ===================== 主函数 =====================
def main():
    split_data = {}
    for s in SPLITS:
        p = os.path.join(ANN_DIR, f"{s}.json")
        if os.path.exists(p):
            split_data[s] = load_json(p)

    all_images = [im for s in split_data for im in split_data[s]["images"]]
    all_annotations = [a for s in split_data for a in split_data[s]["annotations"]]
    all_categories = split_data[next(iter(split_data.keys()))]["categories"] if split_data else []

    full_coco = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": all_categories
    }

    print("==============================================")
    print("                Overall Statistics")
    print("==============================================")
    print_stats("All Data", *count_stats(full_coco))

    print("\n\n==============================================")
    print("                Statistics per Split")
    print("==============================================")
    for s in SPLITS:
        if s in split_data:
            print_stats(f"Split {s}", *count_stats(split_data[s]))

    analyze_by_location(full_coco)
    plot_scientific_stats(full_coco, split_data)

    print("\nAnalysis finished!")

if __name__ == "__main__":
    main()
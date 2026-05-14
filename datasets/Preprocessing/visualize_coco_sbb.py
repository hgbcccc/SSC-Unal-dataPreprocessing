import argparse
import json
import os
import random


#python datasets\Preprocessing\visualize_coco_sbb.py --images-dir datasets\Preprocessing\processed_sbb_splits\unlabeled --ann datasets\Preprocessing\processed_sbb_splits\annotations\unlabeled.json -n 20 --out datasets\Preprocessing\vis_unlabeled

COLORS = {
    "Spruce": (255, 0, 0),    # 云杉 红
    "Pine": (0, 255, 0),      # 松树 绿
    "Birch": (0, 128, 255),   # 桦树 蓝
    "Aspen": (255, 165, 0),   # 白杨 橙
    "Other": (160, 32, 240),  # 其他树种 紫
}


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cat = {c["id"]: c["name"] for c in coco.get("categories", [])}
    anns_by_img = {}
    for a in coco.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)
    return coco, cat, anns_by_img


def draw_one(img_path, anns, cat_map, out_path):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(img_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for a in anns:
        name = cat_map.get(a["category_id"], "unknown")
        r, g, b = COLORS.get(name, (255, 255, 0))
        x, y, w, h = a["bbox"]
        x2, y2 = x + w, y + h
        dr.rectangle([x, y, x2, y2], outline=(r, g, b), width=7)
        dr.text((x + 2, max(0, y - 16)), name, fill=(r, g, b), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images-dir",
        required=True,
        help="图片所在目录（支持两种结构：split/ 或 split/images/）。例如 processed_sbb_splits/val",
    )
    ap.add_argument("--ann", required=True, help="COCO json，例如 processed_sbb_splits/annotations/val.json")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260504)
    ap.add_argument("--out", default="datasets/Preprocessing/vis_out")
    args = ap.parse_args()

    coco, cat_map, anns_by_img = load_coco(args.ann)
    images = coco.get("images", [])
    if not images:
        raise SystemExit("No images in coco json.")

    img_root = args.images_dir
    if os.path.isdir(os.path.join(img_root, "images")):
        img_root = os.path.join(img_root, "images")

    rnd = random.Random(args.seed)
    pick = images if len(images) <= args.n else rnd.sample(images, args.n)
    for im in pick:
        fn = os.path.basename(im["file_name"])
        img_path = os.path.join(img_root, fn)
        if not os.path.isfile(img_path):
            print(f"Skip missing: {img_path}")
            continue
        out_path = os.path.join(args.out, fn)
        draw_one(img_path, anns_by_img.get(im["id"], []), cat_map, out_path)

    print(f"Done. Wrote to: {args.out}")


if __name__ == "__main__":
    main()

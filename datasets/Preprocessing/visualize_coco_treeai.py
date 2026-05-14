import argparse
import json
import os
import random
import re
# python datasets\Preprocessing\visualize_coco_treeai.py --images-dir datasets\Preprocessing\processed_treeai_splits\val --ann datasets\Preprocessing\processed_treeai_splits\annotations\val.json -n 100 --out datasets\Preprocessing\vis_treeai_val 

# TreeAI v1.2 abbreviations used in template.tex:
# AS, LD, PA, PR, FG, BA, AP, PK, PP, PB
COLORS = {
    # High-contrast, easy-to-distinguish palette (avoid very dark tones)
    "AS": (255, 59, 48),    # vivid red
    "LD": (52, 199, 89),    # vivid green
    "PA": (0, 122, 255),    # vivid blue
    "PR": (255, 149, 0),    # orange
    "FG": (175, 82, 222),   # purple
    "BA": (90, 200, 250),   # sky blue
    "AP": (255, 45, 85),    # hot pink
    "PK": (255, 204, 0),    # yellow
    "PP": (255, 105, 180),  # pink
    "PB": (0, 206, 209),    # turquoise
}

def to_abbr(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return "UNK"
    if s in COLORS:
        return s
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split(" ")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return (parts[0][0] + parts[1][0]).upper()
    return (parts[0][:2] if parts else "UNK").upper()


def color_for(label: str):
    if label in COLORS:
        return COLORS[label]
    # deterministic fallback to avoid "all same color"
    h = 0
    for ch in label:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return ((h >> 16) & 255, (h >> 8) & 255, h & 255)


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
        full_name = cat_map.get(a["category_id"], "unknown")
        name = to_abbr(full_name)
        r, g, b = color_for(name)
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
        help="图片所在目录（支持 split/ 或 split/images/）。例如 datasets/Preprocessing/processed_treeai_splits/val",
    )
    ap.add_argument("--ann", required=True, help="COCO json，例如 .../annotations/val.json")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260505)
    ap.add_argument("--out", default="datasets/Preprocessing/vis_treeai_out")
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

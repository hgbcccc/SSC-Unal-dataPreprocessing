import json
import os
import random
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET


# ===================== Split Config (edit here) =====================
SEED = 20260505
SPLIT_RATIOS = {"init_train": 0.05, "val": 0.10, "test": 0.10, "unlabeled": 0.75}
VAL_EQUALS_TEST = False
OUTPUT_DIRNAME = "TreeAI v1.2"
TOPK = 10
# ================================================================


def read_classes_order(path):
    return [int(x.strip()) for x in open(path, "r", encoding="utf-8") if x.strip()]


def read_xlsx_class_table(xlsx_path):
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(xlsx_path)
    ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
    strings = ["".join(t.text or "" for t in si.findall(".//s:t", ns)) for si in ss.findall(".//s:si", ns)]
    sh = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    def cell_text(c):
        v = c.find("s:v", ns)
        if v is None:
            return None
        return strings[int(v.text)] if c.get("t") == "s" else v.text

    rows = sh.findall(".//s:sheetData/s:row", ns)
    data = []
    for r in rows:
        row = [cell_text(c) for c in r.findall("s:c", ns)]
        if row and row[0] and row[0].isdigit() and len(row) >= 3:
            data.append((int(row[0]), int(row[1]), row[2]))
    return data  # (class_id, labels, class_name)


def yolo_to_coco_bbox(line, w, h):
    t = line.strip().split()
    if len(t) < 5:
        return None
    cls_idx = int(float(t[0]))
    cx, cy, bw, bh = map(float, t[1:5])
    x = (cx - bw / 2) * w
    y = (cy - bh / 2) * h
    return cls_idx, [x, y, bw * w, bh * h]


def safe_link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.exists(dst):
            return
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def main():
    base = os.path.join("datasets", "12_RGB_FullyLabeled_640", "coco")
    xlsx = os.path.join(base, "class12_RGB_all_L.xlsx")
    classes_txt = os.path.join(base, "classes.txt")
    out_root = os.path.join("datasets", "Preprocessing", OUTPUT_DIRNAME)
    out_ann = os.path.join(out_root, "annotations")
    os.makedirs(out_ann, exist_ok=True)

    order = read_classes_order(classes_txt)  # label idx -> class_id
    table = read_xlsx_class_table(xlsx)  # (class_id, labels, name)
    table_by_id = {cid: (labels, name) for cid, labels, name in table}
    top_ids = [cid for cid, _, _ in sorted(table, key=lambda x: -x[1])[:TOPK]]
    top_set = set(top_ids)
    categories = [{"id": cid, "name": table_by_id[cid][1]} for cid in top_ids]

    images, annotations = [], []
    img_id, ann_id = 1, 1
    id_map = {}  # stem -> img_id

    from PIL import Image

    for sp in ("train", "val"):
        img_dir = os.path.join(base, sp, "images")
        lab_dir = os.path.join(base, sp, "labels")
        for fn in sorted(os.listdir(img_dir)):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                continue
            stem = os.path.splitext(fn)[0]
            lab_path = os.path.join(lab_dir, stem + ".txt")
            if not os.path.isfile(lab_path):
                continue
            w, h = Image.open(os.path.join(img_dir, fn)).size
            images.append({"id": img_id, "file_name": fn, "width": w, "height": h, "source_split": sp})
            id_map[(sp, stem)] = img_id

            kept = 0
            for line in open(lab_path, "r", encoding="utf-8"):
                r = yolo_to_coco_bbox(line, w, h)
                if not r:
                    continue
                cls_idx, bbox = r
                if cls_idx < 0 or cls_idx >= len(order):
                    continue
                cls_id = order[cls_idx]
                if cls_id not in top_set:
                    continue
                x, y, bw, bh = bbox
                if bw <= 0 or bh <= 0:
                    continue
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1
                kept += 1
            if kept > 0:
                img_id += 1
            else:
                images.pop()  # drop images without top10 boxes

    full = {"images": images, "annotations": annotations, "categories": categories}
    ids = [im["id"] for im in full["images"]]
    random.Random(SEED).shuffle(ids)
    n = len(ids)
    n_init = int(n * SPLIT_RATIOS["init_train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    n_test = 0 if VAL_EQUALS_TEST else int(n * SPLIT_RATIOS["test"])
    n_unl = int(n * SPLIT_RATIOS["unlabeled"])
    init_ids = ids[:n_init]
    val_ids = ids[n_init : n_init + n_val]
    if VAL_EQUALS_TEST:
        test_ids, start = list(val_ids), n_init + n_val
    else:
        test_ids, start = ids[n_init + n_val : n_init + n_val + n_test], n_init + n_val + n_test
    unl_ids = ids[start : start + n_unl]
    used = set(init_ids) | set(val_ids) | set(test_ids) | set(unl_ids)
    rest_ids = [i for i in ids if i not in used]
    splits = {"init_train": init_ids, "val": val_ids, "test": test_ids, "unlabeled": unl_ids, "rest": rest_ids}

    # Merge rest (rounding remainder) into unlabeled pool
    n_rest = len(splits["rest"])
    if n_rest > 0:
        splits["unlabeled"].extend(splits["rest"])
        print(f"[Split] {n_rest} remainder image(s) from 'rest' merged into 'unlabeled'.")
    del splits["rest"]

    total = len(full["images"])
    print(
        f"[Split] total={total} -> "
        f"init_train={len(splits['init_train'])} ({len(splits['init_train'])/total*100:.1f}%), "
        f"val={len(splits['val'])} ({len(splits['val'])/total*100:.1f}%), "
        f"test={len(splits['test'])} ({len(splits['test'])/total*100:.1f}%), "
        f"unlabeled={len(splits['unlabeled'])} ({len(splits['unlabeled'])/total*100:.1f}%)"
    )

    tmp = os.path.join(out_root, "_tmp_all_images")
    os.makedirs(tmp, exist_ok=True)
    print(f"[Staging] Copying source images to temp directory...")
    for sp in ("train", "val"):
        img_dir = os.path.join(base, sp, "images")
        for fn in os.listdir(img_dir):
            if os.path.splitext(fn)[1].lower() in (".jpg", ".jpeg", ".png"):
                safe_link_or_copy(os.path.join(img_dir, fn), os.path.join(tmp, fn))

    print("[Staging] Distributing images into split directories & writing JSON...")
    for name in ("init_train", "val", "test", "unlabeled"):
        keep_ids = splits[name]
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        keep = set(keep_ids)
        sub_imgs = [im.copy() for im in full["images"] if im["id"] in keep]
        sub_anns = [a for a in full["annotations"] if a["image_id"] in keep]
        for im in sub_imgs:
            fn = os.path.basename(im["file_name"])
            safe_link_or_copy(os.path.join(tmp, fn), os.path.join(out_dir, fn))
            im["file_name"] = (os.path.join(name, fn)).replace("\\", "/")
        json.dump(
            {"images": sub_imgs, "annotations": sub_anns, "categories": categories},
            open(os.path.join(out_ann, f"{name}.json"), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )

    shutil.rmtree(tmp, ignore_errors=True)
    print(
        f"[Done] Dataset: {OUTPUT_DIRNAME}\n"
        f"  Top-{TOPK} classes: {top_ids}\n"
        f"  Images: {len(full['images'])}  |  Annotations: {len(full['annotations'])}\n"
        f"  Splits -> init_train: {len(splits['init_train'])}, val: {len(splits['val'])}, "
        f"test: {len(splits['test'])}, unlabeled: {len(splits['unlabeled'])}\n"
        f"  Output: {out_root}"
    )


if __name__ == "__main__":
    main()


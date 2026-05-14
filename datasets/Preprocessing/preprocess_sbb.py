import json
import os
import re
import shutil
import xml.etree.ElementTree as ET


# ===================== Split Config (edit here) =====================
# Ratios are applied after filtering out `damage/`.
# "unlabeled" here means "held out for active learning pool" (still has GT labels).
SEED = 20260504
SPLIT_RATIOS = {
    "init_train": 0.05,  # initial labeled training set
    "val": 0.10,
    "test": 0.10,
    "unlabeled": 0.75,
}
# If True: test set == val set (same images/annotations).
VAL_EQUALS_TEST = False
OUTPUT_DIRNAME = "Spruce-Bark-Beetle v1.0"
# ================================================================

LOC_ABBR = {"Backsjon": "BKS", "Lidhem": "LDH", "Viken": "VKN"}
VIEW_ABBR = {"vertical": "V", "oblique": "O"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_meta(folder_name: str, view_dir: str):
    parts = folder_name.split("_")
    loc = parts[0]
    date = parts[1] if len(parts) > 1 and re.fullmatch(r"\d{8}", parts[1]) else "UNKNOWNDATE"
    view = VIEW_ABBR.get(view_dir.lower())
    if not view:
        return None
    return LOC_ABBR.get(loc, loc[:3].upper()), date, view


def uuid_prefix(stem: str):
    p = stem.split("-")
    return f"{p[0]}-{p[1]}" if len(p) >= 2 else stem


def voc_to_boxes(xml_path: str):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.findtext("width", "0")) if size is not None else 0
    h = int(size.findtext("height", "0")) if size is not None else 0
    items = []
    for obj in root.findall("object"):
        label = (obj.findtext("tree") or obj.findtext("name") or "unknown").strip()
        b = obj.find("bndbox")
        if b is None:
            continue
        xmin = int(float(b.findtext("xmin", "0")))
        ymin = int(float(b.findtext("ymin", "0")))
        xmax = int(float(b.findtext("xmax", "0")))
        ymax = int(float(b.findtext("ymax", "0")))
        bw, bh = max(0, xmax - xmin), max(0, ymax - ymin)
        if bw == 0 or bh == 0:
            continue
        items.append((label, xmin, ymin, bw, bh))
    return w, h, items


def collect_records(root: str, map_path: str):
    records, counters = [], {}
    with open(map_path, "w", encoding="utf-8") as m:
        for view_dir in ("oblique", "vertical"):
            vpath = os.path.join(root, view_dir)
            if not os.path.isdir(vpath):
                continue
            for folder in sorted(os.listdir(vpath)):
                fpath = os.path.join(vpath, folder)
                if not os.path.isdir(fpath) or "damage" in folder.lower():
                    continue
                meta = parse_meta(folder, view_dir)
                if not meta:
                    continue
                loc, date, view = meta
                img_dir = os.path.join(fpath, "Images")
                ann_dir = os.path.join(fpath, "Annotations")
                if not (os.path.isdir(img_dir) and os.path.isdir(ann_dir)):
                    continue
                key = (loc, date, view)
                counters.setdefault(key, 0)
                for fn in sorted(os.listdir(img_dir)):
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in IMG_EXTS:
                        continue
                    stem = os.path.splitext(fn)[0]
                    xml_path = os.path.join(ann_dir, stem + ".xml")
                    if not os.path.isfile(xml_path):
                        continue
                    counters[key] += 1
                    seq = f"{counters[key]:06d}"
                    new_fn = f"{loc}_{date}_{view}_{seq}{ext}"
                    m.write(f"{stem}\t{os.path.splitext(new_fn)[0]}\t{uuid_prefix(stem)}\n")
                    records.append(
                        {
                            "old_stem": stem,
                            "new_fn": new_fn,
                            "old_img": os.path.join(img_dir, fn),
                            "xml_path": xml_path,
                            "loc": loc,
                            "date": date,
                            "view": view,
                            "source_folder": folder,
                            "orig_prefix": uuid_prefix(stem),
                        }
                    )
    return records


def build_full_coco(records):
    coco = {"images": [], "annotations": [], "categories": []}
    cats, next_cat_id = {}, 1
    img_id, ann_id = 1, 1
    for r in records:
        w, h, boxes = voc_to_boxes(r["xml_path"])
        coco["images"].append(
            {
                "id": img_id,
                "file_name": r["new_fn"],  # 直接文件名，无路径
                "width": w,
                "height": h,
                "loc": r["loc"],
                "date": r["date"],
                "view": r["view"],
                "orig_prefix": r["orig_prefix"],
                "old_stem": r["old_stem"],
                "source_folder": r["source_folder"],
            }
        )
        for label, x, y, bw, bh in boxes:
            if label not in cats:
                cats[label] = next_cat_id
                next_cat_id += 1
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cats[label],
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1
        img_id += 1
    coco["categories"] = [{"id": cid, "name": name} for name, cid in sorted(cats.items(), key=lambda x: x[1])]
    return coco


def split_image_ids(images, ratios, seed, val_equals_test):
    import random

    ids = [im["id"] for im in images]
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_init = int(n * ratios["init_train"])
    n_val = int(n * ratios["val"])
    n_test = 0 if val_equals_test else int(n * ratios["test"])
    n_unl = int(n * ratios["unlabeled"])

    init_ids = ids[:n_init]
    val_ids = ids[n_init : n_init + n_val]
    if val_equals_test:
        test_ids = list(val_ids)
        start = n_init + n_val
    else:
        test_ids = ids[n_init + n_val : n_init + n_val + n_test]
        start = n_init + n_val + n_test
    unl_ids = ids[start : start + n_unl]
    used = set(init_ids) | set(val_ids) | set(test_ids) | set(unl_ids)
    rest_ids = [i for i in ids if i not in used]
    return {"init_train": init_ids, "val": val_ids, "test": test_ids, "unlabeled": unl_ids, "rest": rest_ids}


def subset_coco(full, keep_ids):
    keep = set(keep_ids)
    images = [im for im in full["images"] if im["id"] in keep]
    ann = [a for a in full["annotations"] if a["image_id"] in keep]
    return {"images": images, "annotations": ann, "categories": full["categories"]}


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.exists(dst):
            return
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


# 直接保存到子集目录，不套 images
def stage_split_images(full, out_root, split_name, image_ids, src_all_dir, use_dir=None):
    split_dir = use_dir or os.path.join(out_root, split_name)
    os.makedirs(split_dir, exist_ok=True)
    keep = set(image_ids)
    for im in full["images"]:
        if im["id"] not in keep:
            continue
        fn = im["file_name"]
        safe_link_or_copy(os.path.join(src_all_dir, fn), os.path.join(split_dir, fn))
    return split_dir


# 重写：直接使用文件名，无 images 路径
def rewrite_filenames(coco_obj, split_name):
    for im in coco_obj["images"]:
        im["file_name"] = im["file_name"]  # 保持纯文件名
    return coco_obj


def main():
    root = os.path.join("datasets", "Data_Set_Spruce_Bark_Beetle")
    out_root = os.path.join("datasets", "Preprocessing", OUTPUT_DIRNAME)
    out_ann = os.path.join(out_root, "annotations")
    os.makedirs(out_ann, exist_ok=True)

    map_path = os.path.join("datasets", "Preprocessing", "name_map.txt")
    records = collect_records(root, map_path)
    full = build_full_coco(records)
    splits = split_image_ids(full["images"], SPLIT_RATIOS, SEED, VAL_EQUALS_TEST)

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

    tmp_all_dir = os.path.join(out_root, "_tmp_all_images")
    os.makedirs(tmp_all_dir, exist_ok=True)
    print(f"[Staging] Copying {len(records)} images to temp directory...")
    for r in records:
        safe_link_or_copy(r["old_img"], os.path.join(tmp_all_dir, r["new_fn"]))

    print("[Staging] Distributing images into split directories...")
    val_dir = stage_split_images(full, out_root, "val", splits["val"], tmp_all_dir)
    for name in ("init_train", "unlabeled"):
        stage_split_images(full, out_root, name, splits[name], tmp_all_dir)
    if VAL_EQUALS_TEST:
        stage_split_images(full, out_root, "test", splits["test"], tmp_all_dir, use_dir=val_dir)
    else:
        stage_split_images(full, out_root, "test", splits["test"], tmp_all_dir)

    print("[JSON] Writing COCO annotation files...")
    for name in ("init_train", "val", "test", "unlabeled"):
        obj = subset_coco(full, splits[name])
        write_json(os.path.join(out_ann, f"{name}.json"), rewrite_filenames(obj, name))

    shutil.rmtree(tmp_all_dir, ignore_errors=True)

    print(
        f"[Done] Dataset: {OUTPUT_DIRNAME}\n"
        f"  Images: {len(full['images'])}  |  Annotations: {len(full['annotations'])}  |  Categories: {len(full['categories'])}\n"
        f"  Splits -> init_train: {len(splits['init_train'])}, val: {len(splits['val'])}, "
        f"test: {len(splits['test'])}, unlabeled: {len(splits['unlabeled'])}\n"
        f"  Output: {out_root}"
    )


if __name__ == "__main__":
    main()

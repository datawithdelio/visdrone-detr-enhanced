import json
from pathlib import Path
from PIL import Image

# VisDrone classes (commonly used)
# 0=ignored regions, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van, 6=truck,
# 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others
VISDRONE_ID_TO_NAME = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

def read_visdrone_txt(txt_path: Path):
    """
    Each line:
    x, y, w, h, score, class_id, truncation, occlusion
    For train/val, score is usually 1; we ignore it.
    """
    boxes = []
    with txt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            x, y, w, h = map(float, parts[0:4])
            class_id = int(parts[5])

            # Skip ignored regions or invalid boxes
            if class_id == 0:
                continue
            if w <= 0 or h <= 0:
                continue

            boxes.append((x, y, w, h, class_id))
    return boxes

def convert_split(visdrone_root: Path, split_name: str, out_json: Path):
    images_dir = visdrone_root / split_name / "images"
    ann_dir = visdrone_root / split_name / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Missing annotations dir: {ann_dir}")

    categories = []
    cat_id_map = {}  # visdrone class_id -> coco category id

    coco_cat_id = 1
    for vid, name in VISDRONE_ID_TO_NAME.items():
        cat_id_map[vid] = coco_cat_id
        categories.append({"id": coco_cat_id, "name": name, "supercategory": "object"})
        coco_cat_id += 1

    coco = {"images": [], "annotations": [], "categories": categories}

    img_id = 1
    ann_id = 1

    # VisDrone filenames are like 0000001.jpg and annotations 0000001.txt
    img_paths = sorted(images_dir.glob("*.jpg"))
    if not img_paths:
        # sometimes .png
        img_paths = sorted(images_dir.glob("*.png"))

    for img_path in img_paths:
        stem = img_path.stem
        txt_path = ann_dir / f"{stem}.txt"
        if not txt_path.exists():
            # If annotation missing, skip
            continue

        # Read image size
        with Image.open(img_path) as im:
            width, height = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })

        boxes = read_visdrone_txt(txt_path)
        for x, y, w, h, class_id in boxes:
            # COCO bbox is [x,y,w,h]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id_map[class_id],
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco))
    print(f"Saved {out_json} | images={len(coco['images'])} anns={len(coco['annotations'])}")

def main():
    # Adjust this root if needed
    root = Path("/Users/deliorincon/Desktop/Kumar")

    # We’ll point to your extracted VisDrone folders and create a COCO folder structure
    # Expected:
    # /Users/.../Desktop/Kumar/VisDrone2019-DET-train/images, annotations
    # /Users/.../Desktop/Kumar/VisDrone2019-DET-val/images, annotations
    visdrone_root = root

    # Create normalized split folders: train/val under root/visdrone/
    # We’ll symlink/copy later if you want, but for now we convert directly from your current folders.
    # So we map:
    # split "VisDrone2019-DET-train" -> output instances_train.json
    # split "VisDrone2019-DET-val"   -> output instances_val.json

    convert_split(visdrone_root, "VisDrone2019-DET-train", root / "coco/annotations/instances_train.json")
    convert_split(visdrone_root, "VisDrone2019-DET-val", root / "coco/annotations/instances_val.json")

if __name__ == "__main__":
    main()

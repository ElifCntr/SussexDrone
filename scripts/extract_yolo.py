"""SussexDrone YOLO extractor.

Converts the SussexDrone dataset (Drone-vs-Bird format: videos + per-video
annotation files + split lists) into YOLO-format training data: extracted
frames as JPEG images plus per-frame label files, organised into
train/val/test folders, with a data.yaml ready for Ultralytics YOLO.

This script operates on the published dataset as downloaded from Figshare.
It does not require any internal tooling.

------------------------------------------------------------------------
Expected input layout (as published):

    <dataset_root>/
        videos/                 e.g. dji_grass_001.mp4
        annotations/            e.g. dji_grass_001.txt
        splits/
            detection_train.txt    detection_val.txt    detection_test.txt
            trajectory_train.txt   trajectory_val.txt   trajectory_test.txt

Annotation format (one line per video frame):
    frame_index  num_objects  [x_left y_top width height class] ...
Coordinates are absolute pixels; class is "drone" or "bird".
A line "12 0" means frame 12 contains no objects.

------------------------------------------------------------------------
Output layout (Ultralytics-compatible):

    <output_root>/
        images/train/  images/val/  images/test/
        labels/train/  labels/val/  labels/test/
        data.yaml

YOLO label format (normalised, one row per object):
    class_id  x_centre  y_centre  width  height      (all in 0..1)

------------------------------------------------------------------------
Usage:

    pip install opencv-python pyyaml tqdm

    python extract_yolo.py --dataset /path/to/SussexDrone \\
                           --output  /path/to/SussexDrone_yolo \\
                           --subset  detection

Options:
    --subset {detection,trajectory}   which split lists to use (default detection)
    --skip-empty                      drop frames with no objects
    --jpeg-quality N                  JPEG quality 1-100 (default 95)
    --classes drone bird              class order; index = position (default: drone bird)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op
    def tqdm(x, **kwargs):
        return x


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_annotation_file(path: Path, class_to_id: dict[str, int]) -> dict[int, list]:
    """Parse a Drone-vs-Bird annotation file.

    Returns {frame_index: [(class_id, x_left, y_top, width, height), ...]}.
    Frames with no objects map to an empty list.
    """
    frames: dict[int, list] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            frame_idx = int(parts[0])
            n_obj = int(parts[1])
        except (ValueError, IndexError):
            raise ValueError(f"{path.name}: malformed line {line_no}: {line!r}")
        boxes = []
        for i in range(n_obj):
            base = 2 + i * 5
            x, y, w, h = (int(parts[base]), int(parts[base + 1]),
                          int(parts[base + 2]), int(parts[base + 3]))
            cls_name = parts[base + 4]
            if cls_name not in class_to_id:
                # Class not in the requested set; skip this box silently.
                continue
            boxes.append((class_to_id[cls_name], x, y, w, h))
        frames[frame_idx] = boxes
    return frames


def to_yolo_line(class_id: int, x: int, y: int, w: int, h: int,
                 img_w: int, img_h: int) -> str | None:
    """Convert an absolute pixel box to a normalised YOLO label line.

    Clips to image bounds; returns None for a degenerate (zero-area) box.
    """
    x1 = max(0, min(x, img_w))
    y1 = max(0, min(y, img_h))
    x2 = max(0, min(x + w, img_w))
    y2 = max(0, min(y + h, img_h))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    nw = bw / img_w
    nh = bh / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


# ---------------------------------------------------------------------------
# Split handling
# ---------------------------------------------------------------------------

def read_split(split_path: Path) -> list[str]:
    """Read a split file; return list of video filenames (without path)."""
    if not split_path.exists():
        return []
    return [ln.strip() for ln in split_path.read_text().splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_video(video_path: Path, ann_path: Path, split: str,
                  images_root: Path, labels_root: Path,
                  class_to_id: dict[str, int],
                  skip_empty: bool, jpeg_quality: int) -> tuple[int, int]:
    """Extract all frames of one video to YOLO images + labels.

    Returns (frames_written, boxes_written).
    """
    annotations = parse_annotation_file(ann_path, class_to_id)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path.name}, skipping")
        return 0, 0

    img_dir = images_root / split
    lbl_dir = labels_root / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    frames_written = 0
    boxes_written = 0
    frame_idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        boxes = annotations.get(frame_idx, [])
        if skip_empty and not boxes:
            frame_idx += 1
            continue
        img_h, img_w = img.shape[:2]
        label_lines = []
        for (cls_id, x, y, w, h) in boxes:
            line = to_yolo_line(cls_id, x, y, w, h, img_w, img_h)
            if line is not None:
                label_lines.append(line)
        # Write image + label (label may be empty = background frame)
        name = f"{stem}_f{frame_idx:06d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        (lbl_dir / f"{name}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
        frames_written += 1
        boxes_written += len(label_lines)
        frame_idx += 1

    cap.release()
    return frames_written, boxes_written


def main():
    ap = argparse.ArgumentParser(description="Extract SussexDrone to YOLO format.")
    ap.add_argument("--dataset", required=True, help="Path to SussexDrone dataset root")
    ap.add_argument("--output", required=True, help="Where to write YOLO data")
    ap.add_argument("--subset", choices=["detection", "trajectory"], default="detection",
                    help="Which split lists to use (default: detection)")
    ap.add_argument("--skip-empty", action="store_true",
                    help="Drop frames with no objects (default: keep as background)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--classes", nargs="+", default=["drone", "bird"],
                    help="Class order; list index is the YOLO id (default: drone bird)")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    videos_dir = dataset / "videos"
    ann_dir = dataset / "annotations"
    splits_dir = dataset / "splits"

    for d in (videos_dir, ann_dir, splits_dir):
        if not d.exists():
            sys.exit(f"ERROR: expected folder not found: {d}")

    class_to_id = {name: i for i, name in enumerate(args.classes)}
    print(f"Classes: {class_to_id}")
    print(f"Subset: {args.subset}")
    print(f"Empty frames: {'dropped' if args.skip_empty else 'kept as background'}")

    images_root = output / "images"
    labels_root = output / "labels"

    total_frames = 0
    total_boxes = 0
    split_counts = {}
    for split in ("train", "val", "test"):
        split_file = splits_dir / f"{args.subset}_{split}.txt"
        videos = read_split(split_file)
        if not videos:
            print(f"\n[{split}] no videos in {split_file.name}, skipping")
            continue
        print(f"\n[{split}] {len(videos)} videos")
        sf = sb = 0
        for vid in tqdm(videos, desc=f"  {split}", unit="vid"):
            video_path = videos_dir / vid
            ann_path = ann_dir / (Path(vid).stem + ".txt")
            if not video_path.exists():
                print(f"  WARNING: missing video {vid}")
                continue
            if not ann_path.exists():
                print(f"  WARNING: missing annotation for {vid}")
                continue
            fw, bw = extract_video(video_path, ann_path, split,
                                   images_root, labels_root, class_to_id,
                                   args.skip_empty, args.jpeg_quality)
            sf += fw
            sb += bw
        split_counts[split] = sf
        total_frames += sf
        total_boxes += bw if False else sb  # keep readable
        print(f"  {split}: {sf:,} frames, {sb:,} boxes")

    # Write data.yaml for Ultralytics
    output.mkdir(parents=True, exist_ok=True)
    names_block = "\n".join(f"  {i}: {name}" for name, i in class_to_id.items())
    data_yaml = (
        f"# SussexDrone — YOLO data config ({args.subset} subset)\n"
        f"path: {output.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n{names_block}\n"
    )
    (output / "data.yaml").write_text(data_yaml)

    print("\n" + "=" * 50)
    print(f"Done. {total_frames:,} frames, {total_boxes:,} boxes written to {output}")
    print(f"data.yaml written. Train with e.g.:")
    print(f"  yolo detect train data={output / 'data.yaml'} model=yolov8n.pt epochs=100")


if __name__ == "__main__":
    main()

"""Build a SketchNet training set from Rico.

Rico ships Android screenshots plus view hierarchies. This turns a subset of
them into COCO-format object detection annotations: one PNG per screen and a
single instances.json naming every visible UI widget's box and type.

Two things about Rico's schema are easy to get wrong, and both silently
corrupt the labels rather than raising:

1. `activity["children"]` is NOT a recursive tree. It is a flat list of depth
   levels ordered leaf-first, where every field of a level is a parallel list
   across the nodes at that depth. So level i has len(level["klass"]) nodes.

2. Bounds are in device coordinates on a fixed 1440x2560 screen, while the
   screenshots are 1080x1920 or 540x960. Boxes must be rescaled. Scaling by
   the root node's own width is tempting and wrong: on roughly 8% of screens
   the root does not span the full 1440, which quietly skews every box.

KNOWN LIMITATION of this HF config: the flattening is lossy. A real Rico
screen has dozens of widgets, but `children` carries only the top of the
tree (17 nodes on screen 0), so leaf widgets — the TextViews, EditTexts and
ImageViews under a RadioGroup or TextInputLayout — are simply absent. In
practice this yields ~1.7 boxes/screen, almost all navbar and list, with no
text/input/image at all. That is not enough to train a component detector.
Getting full hierarchies means going to Rico's original JSON distribution
rather than this HuggingFace config.

Run: python -m data.build_dataset
"""

import json
import os
from collections import Counter

from datasets import load_dataset

OUTPUT_IMAGES = "data/rico/raw"
OUTPUT_ANNOTATIONS = "data/annotations/instances.json"

NUM_SAMPLES = 3000

# Rico's device coordinate space. Bounds are always expressed against this,
# regardless of the resolution the screenshot was saved at.
DEVICE_WIDTH = 1440
DEVICE_HEIGHT = 2560

# Boxes thinner than this (in screenshot pixels) after scaling are dropped.
MIN_BOX_SIDE = 8

# Android widget class -> SketchNet component type. Matched against the last
# dotted segment of the class name, lowercased. Pure layout containers
# (LinearLayout, FrameLayout, RelativeLayout...) are deliberately absent:
# they are structure, not components, and labelling them buries the real
# widgets under giant overlapping boxes.
CLASS_MAP = [
    ("edittext", "input"),
    ("autocompletetextview", "input"),
    ("searchview", "input"),
    ("button", "button"),
    ("checkbox", "toggle"),
    ("radiobutton", "toggle"),
    ("switch", "toggle"),
    ("togglebutton", "toggle"),
    ("seekbar", "slider"),
    ("progressbar", "progress"),
    ("toolbar", "navbar"),
    ("actionbar", "navbar"),
    ("actionmenuview", "navbar"),
    ("tablayout", "tabs"),
    ("cardview", "card"),
    ("recyclerview", "list"),
    ("listview", "list"),
    ("gridview", "list"),
    ("imageview", "image"),
    ("imagebutton", "button"),
    ("textview", "text"),
]

CATEGORIES = sorted({name for _, name in CLASS_MAP})
CATEGORY_ID = {name: i + 1 for i, name in enumerate(CATEGORIES)}


def component_type(klass):
    """Map an Android class name to a component type, or None to skip it."""

    leaf = klass.rsplit(".", 1)[-1].lower()

    # Longest pattern first so "imagebutton" beats "button" and
    # "autocompletetextview" beats "textview".
    for pattern, name in sorted(CLASS_MAP, key=lambda p: -len(p[0])):
        if pattern in leaf:
            return name

    return None


def iter_nodes(activity):
    """Yield every node in the hierarchy as a flat dict of its own fields."""

    root = activity["root"]

    yield {key: root[key] for key in root}

    for level in activity["children"]:

        count = len(level["klass"])

        for i in range(count):
            yield {key: values[i] for key, values in level.items()}


def scale_box(bounds, scale, img_w, img_h):
    """Device-space [x1,y1,x2,y2] -> screenshot-space [x,y,w,h], clipped."""

    x1, y1, x2, y2 = (v * scale for v in bounds)

    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))

    w = x2 - x1
    h = y2 - y1

    if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
        return None

    return [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)]


def extract(example, image_id):
    """Return (annotations, skipped_count) for one screen."""

    img_w, img_h = example["screenshot"].size

    # Uniform scale from the fixed device width, NOT from the root's bounds.
    scale = img_w / DEVICE_WIDTH

    annotations = []
    seen = set()
    skipped = 0

    for node in iter_nodes(example["activity"]):

        if not node.get("visible_to_user"):
            skipped += 1
            continue

        if node.get("visibility") != "visible":
            skipped += 1
            continue

        name = component_type(node["klass"])

        if name is None:
            skipped += 1
            continue

        box = scale_box(node["bounds"], scale, img_w, img_h)

        if box is None:
            skipped += 1
            continue

        key = (name, *box)

        if key in seen:
            skipped += 1
            continue

        seen.add(key)

        annotations.append(
            {
                "image_id": image_id,
                "category_id": CATEGORY_ID[name],
                "bbox": box,
                "area": round(box[2] * box[3], 2),
                "iscrowd": 0,
            }
        )

    return annotations, skipped


def main():

    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_ANNOTATIONS), exist_ok=True)

    print("Loading Rico (uses the local HuggingFace cache if present)...")

    dataset = load_dataset(
        "shunk031/Rico",
        name="ui-screenshots-and-view-hierarchies",
        split="train",
    )

    total = min(NUM_SAMPLES, len(dataset))

    print(f"Full split: {len(dataset)} screens. Exporting {total}.")

    images = []
    annotations = []
    label_counts = Counter()
    empty_screens = 0
    next_annotation_id = 1

    for image_id in range(total):

        example = dataset[image_id]
        screenshot = example["screenshot"]

        file_name = f"screen_{image_id:05d}.png"
        screenshot.save(os.path.join(OUTPUT_IMAGES, file_name))

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": screenshot.width,
                "height": screenshot.height,
            }
        )

        found, _ = extract(example, image_id)

        if not found:
            empty_screens += 1

        for annotation in found:
            annotation["id"] = next_annotation_id
            next_annotation_id += 1
            label_counts[annotation["category_id"]] += 1

        annotations.extend(found)

        if image_id % 200 == 0:
            print(f"  {image_id}/{total} screens, {len(annotations)} boxes")

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": CATEGORY_ID[name], "name": name}
            for name in CATEGORIES
        ],
    }

    with open(OUTPUT_ANNOTATIONS, "w") as f:
        json.dump(coco, f)

    print(f"\nWrote {len(images)} images to {OUTPUT_IMAGES}")
    print(f"Wrote {len(annotations)} boxes to {OUTPUT_ANNOTATIONS}")
    print(f"Screens with no usable box: {empty_screens}")
    print("\nBoxes per class:")

    id_to_name = {v: k for k, v in CATEGORY_ID.items()}

    for category_id, count in label_counts.most_common():
        print(f"  {id_to_name[category_id]:10s} {count}")


if __name__ == "__main__":
    main()

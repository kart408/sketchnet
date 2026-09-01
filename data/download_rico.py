"""
Day 1 — Dataset sourcing for SketchNet.

Downloads a subset of Rico (digital UI screenshots + view hierarchies)
and saves it locally for processing in Day 2.

Run: python -m data.download_rico
"""
import os
import json
from datasets import load_dataset

OUTPUT_DIR = "data/rico/raw"
NUM_SAMPLES = 3000  # subsample — full Rico is 66k+ screens, way more than we need

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Rico dataset from HuggingFace (this may take a few minutes)...")
    dataset = load_dataset(
        "shunk031/Rico",
        name="ui-screenshots-and-view-hierarchies",
        split="train",
    )

    print(f"Full dataset size: {len(dataset)} screens")
    print(f"Subsampling to {NUM_SAMPLES} screens...")

    subset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

    for i, example in enumerate(subset):
        # Save screenshot
        image = example["screenshot"]
        image.save(os.path.join(OUTPUT_DIR, f"screen_{i:05d}.png"))

        # Save view hierarchy (contains bounding boxes + component types)
        hierarchy = example["hierarchy"]  # field name may vary — inspect dataset schema first
        with open(os.path.join(OUTPUT_DIR, f"screen_{i:05d}.json"), "w") as f:
            json.dump(hierarchy, f)

        if i % 200 == 0:
            print(f"Saved {i}/{len(subset)}")

    print(f"Done. Saved {len(subset)} screen+hierarchy pairs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

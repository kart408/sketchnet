"""
Day 1 — Inspect the Rico dataset schema before writing the full download script.
Run this first. Paste the printed output back so the loader script can be
adjusted to match the real field names.
"""

from datasets import load_dataset

ds = load_dataset("shunk031/Rico", name="ui-screenshots-and-view-hierarchies", split="train")

print("Number of examples:", len(ds))
print("Keys:", ds[0].keys())
print(ds[0])

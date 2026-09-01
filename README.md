# SketchNet — From Paper Sketch to HTML

> Draw a UI wireframe on paper, take a photo, get rendered HTML in seconds.

## Status

**v1.1 — heuristic baseline working.** OCR + contour detection produce HTML,
React (JSX) and Tailwind from a photo, with no trained model involved.
This is the baseline any future model has to beat.

**Next: dataset sourcing** — pull a Rico subset and build the annotation
pipeline for a learned component detector.

## How it works today

1. **Preprocess** — normalise to grayscale, blur, threshold.
2. **OCR** — Tesseract gives word boxes; keywords map to component types
   (`email`, `password`, `button`).
3. **Shapes** — Canny + `findContours` give boxes; size and aspect ratio map
   to types (`navbar`, `input`, `button`, `card`).
4. **Assemble** — each word attaches to the smallest box enclosing it, so a
   word inside a button labels the button, not the surrounding card. Words
   outside every box become standalone components.
5. **Order** — components are grouped into rows by vertical position, then
   sorted left to right, giving reading order.
6. **Render** — one component list, three renderers (HTML / React / Tailwind).

## Setup

Requires Tesseract on the system:

```bash
brew install tesseract
```

Then:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python app/app.py
```

## Layout

```
app/                 Gradio app + heuristic pipeline
data/download_rico.py  Rico subset downloader
data/rico/raw/       downloaded screens (gitignored)
data/annotations/    processed annotations
train/               training code (empty)
models/              weights (empty, gitignored)
```

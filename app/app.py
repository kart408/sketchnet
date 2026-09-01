import html as html_lib

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import pytesseract


# Boxes whose top edges fall within this many pixels are treated as one row.
ROW_TOLERANCE = 25

# Contours smaller than this are noise, not UI shapes.
MIN_SHAPE_AREA = 500


# ==========================
# PREPROCESSING
# ==========================

def to_rgb_and_gray(image):
    """Normalise any PIL input (L / RGB / RGBA) to an RGB view and a grayscale one."""

    img = np.array(image)

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        gray = img

    elif img.shape[2] == 4:
        rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    else:
        rgb = img
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return rgb, gray


def is_duplicate(box, existing_boxes):
    x, y, w, h = box

    for ex, ey, ew, eh in existing_boxes:
        if (
            abs(x - ex) < 10
            and abs(y - ey) < 10
            and abs(w - ew) < 20
            and abs(h - eh) < 20
        ):
            return True

    return False


# ==========================
# CLASSIFICATION
# ==========================

def classify_shape(w, h):

    aspect_ratio = w / h

    if w > 250 and h < 80:
        return "navbar"

    elif aspect_ratio > 4:
        return "input"

    elif 1.5 < aspect_ratio < 4:
        return "button"

    elif w > 200 and h > 120:
        return "card"

    return "unknown"


def classify_text(text):

    text_upper = text.upper()

    if "EMAIL" in text_upper:
        return "email"

    elif "PASSWORD" in text_upper:
        return "password"

    elif (
        "LOGIN" in text_upper
        or "SIGN IN" in text_upper
        or "SUBMIT" in text_upper
    ):
        return "button"

    return "text"


# ==========================
# COMPONENT ASSEMBLY
# ==========================

def contains(shape, item):
    """True if the centre of an OCR word sits inside a shape's box."""

    cx = item["x"] + item["w"] / 2
    cy = item["y"] + item["h"] / 2

    return (
        shape["x"] <= cx <= shape["x"] + shape["w"]
        and shape["y"] <= cy <= shape["y"] + shape["h"]
    )


def merge_text_into_shapes(words, shapes):
    """Fold each OCR word into the shape that encloses it.

    A word inside a box labels that box (and can refine its type — a box
    holding "EMAIL" is an email input, not a generic one). A word with no
    enclosing box becomes a standalone component of its own.
    """

    components = [dict(shape, text="") for shape in shapes]

    for word in words:

        enclosing = [s for s in components if contains(s, word)]

        # Innermost container wins: a word inside a button inside a card
        # labels the button, not the card.
        host = min(
            enclosing,
            key=lambda s: s["w"] * s["h"],
            default=None
        )

        if host is None:
            components.append(
                dict(word, type=classify_text(word["text"]))
            )
            continue

        host["text"] = f"{host['text']} {word['text']}".strip()

        refined = classify_text(word["text"])

        if refined != "text":
            host["type"] = refined

    return components


def sort_by_layout(components):
    """Reading order: group into rows by vertical position, then left to right."""

    rows = []

    for comp in sorted(components, key=lambda c: c["y"]):

        if rows and abs(comp["y"] - rows[-1][0]["y"]) <= ROW_TOLERANCE:
            rows[-1].append(comp)

        else:
            rows.append([comp])

    ordered = []

    for row in rows:
        ordered.extend(sorted(row, key=lambda c: c["x"]))

    return ordered


# ==========================
# RENDERERS
# ==========================

def label_for(comp, fallback):
    return comp["text"].strip() or fallback


def render_html(components):

    parts = [
        '<div style="max-width:400px;margin:auto;'
        'padding:20px;font-family:Arial;">'
    ]

    for comp in components:

        kind = comp["type"]
        text = html_lib.escape(label_for(comp, ""))

        if kind == "text":
            parts.append(f"  <p>{text}</p>")

        elif kind in ("email", "password", "input"):
            field = "text" if kind == "input" else kind
            placeholder = html_lib.escape(
                label_for(comp, field.capitalize())
            )
            parts.append(
                f'  <input type="{field}" placeholder="{placeholder}" '
                'style="width:100%;padding:12px;margin-bottom:15px;">'
            )

        elif kind == "button":
            parts.append(
                '  <button style="width:100%;padding:12px;'
                f'margin-bottom:15px;">{html_lib.escape(label_for(comp, "Button"))}</button>'
            )

        elif kind == "navbar":
            parts.append(
                '  <nav style="background:#1976d2;color:white;'
                f'padding:15px;margin-bottom:20px;">{html_lib.escape(label_for(comp, "Navbar"))}</nav>'
            )

        elif kind == "card":
            parts.append(
                '  <div style="border:1px solid #ccc;padding:20px;'
                'margin-bottom:15px;border-radius:8px;">'
                f'{html_lib.escape(label_for(comp, "Card"))}</div>'
            )

    parts.append("</div>")

    return "\n".join(parts)


def render_tailwind(components):

    parts = ['<div class="max-w-md mx-auto p-6">']

    for comp in components:

        kind = comp["type"]
        text = html_lib.escape(label_for(comp, ""))

        if kind == "text":
            parts.append(f'  <p class="mb-3">{text}</p>')

        elif kind in ("email", "password", "input"):
            field = "text" if kind == "input" else kind
            placeholder = html_lib.escape(
                label_for(comp, field.capitalize())
            )
            parts.append(
                f'  <input type="{field}" class="w-full border p-2 mb-4" '
                f'placeholder="{placeholder}" />'
            )

        elif kind == "button":
            parts.append(
                '  <button class="w-full bg-blue-600 text-white px-4 py-2 mb-4 rounded">'
                f'{html_lib.escape(label_for(comp, "Button"))}</button>'
            )

        elif kind == "navbar":
            parts.append(
                '  <nav class="bg-blue-600 text-white p-4 mb-5">'
                f'{html_lib.escape(label_for(comp, "Navbar"))}</nav>'
            )

        elif kind == "card":
            parts.append(
                '  <div class="border rounded p-6 mb-4">'
                f'{html_lib.escape(label_for(comp, "Card"))}</div>'
            )

    parts.append("</div>")

    return "\n".join(parts)


def render_react(components):

    body = []

    for comp in components:

        kind = comp["type"]
        text = html_lib.escape(label_for(comp, ""))

        if kind == "text":
            body.append(f'      <p className="mb-3">{text}</p>')

        elif kind in ("email", "password", "input"):
            field = "text" if kind == "input" else kind
            placeholder = html_lib.escape(
                label_for(comp, field.capitalize())
            )
            body.append(
                f'      <input type="{field}" className="w-full border p-2 mb-4" '
                f'placeholder="{placeholder}" />'
            )

        elif kind == "button":
            body.append(
                '      <button className="w-full bg-blue-600 text-white px-4 py-2 mb-4 rounded">'
                f'{html_lib.escape(label_for(comp, "Button"))}</button>'
            )

        elif kind == "navbar":
            body.append(
                '      <nav className="bg-blue-600 text-white p-4 mb-5">'
                f'{html_lib.escape(label_for(comp, "Navbar"))}</nav>'
            )

        elif kind == "card":
            body.append(
                '      <div className="border rounded p-6 mb-4">'
                f'{html_lib.escape(label_for(comp, "Card"))}</div>'
            )

    return (
        "export default function Sketch() {\n"
        "  return (\n"
        '    <div className="max-w-md mx-auto p-6">\n'
        + "\n".join(body)
        + "\n    </div>\n"
        "  );\n"
        "}\n"
    )


# ==========================
# PIPELINE
# ==========================

def detect_words(gray, display_img):

    data = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DICT
    )

    words = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if not text:
            continue

        word = {
            "text": text,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
        }

        words.append(word)

        cv2.rectangle(
            display_img,
            (word["x"], word["y"]),
            (word["x"] + word["w"], word["y"] + word["h"]),
            (255, 0, 0),
            2
        )

    return words


def detect_shapes(blurred, display_img):

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    shapes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if w * h <= MIN_SHAPE_AREA or w <= 10 or h <= 10:
            continue

        if is_duplicate((x, y, w, h), boxes):
            continue

        boxes.append((x, y, w, h))

        shapes.append(
            {
                "type": classify_shape(w, h),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    for i, (x, y, w, h) in enumerate(
        sorted(boxes, key=lambda b: (b[1], b[0]))
    ):

        cv2.rectangle(
            display_img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_img,
            str(i),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return shapes


def sketch_to_html(image):

    if image is None:
        return "", "", "", "<p>Upload a sketch to begin.</p>", None

    rgb, gray = to_rgb_and_gray(image)

    display_img = rgb.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY
    )[1]

    words = detect_words(binary, display_img)
    shapes = detect_shapes(blurred, display_img)

    components = sort_by_layout(
        merge_text_into_shapes(words, shapes)
    )

    rendered = [c for c in components if c["type"] != "unknown"]
    unclassified = len(components) - len(rendered)

    html_code = render_html(rendered)
    react_code = render_react(rendered)
    tailwind_code = render_tailwind(rendered)

    full_text = " ".join(
        w["text"] for w in sort_by_layout(words)
    )

    summary_html = f"""
<h2>SketchNet Analysis</h2>
<p><strong>Shapes detected:</strong> {len(shapes)}</p>
<p><strong>OCR words:</strong> {len(words)}</p>
<p><strong>Components emitted:</strong> {len(rendered)}
({unclassified} unclassified, not rendered)</p>
<h3>OCR Text</h3>
<pre>{html_lib.escape(full_text)}</pre>
<h3>Preview</h3>
"""

    return (
        html_code,
        react_code,
        tailwind_code,
        summary_html + html_code,
        Image.fromarray(display_img),
    )


demo = gr.Interface(
    fn=sketch_to_html,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Code(language="html", label="Generated HTML"),
        gr.Code(language="javascript", label="Generated React"),
        gr.Code(language="html", label="Tailwind Code"),
        gr.HTML(label="Preview"),
        gr.Image(label="Detected Components"),
    ],
    title="SketchNet v1.1",
)


if __name__ == "__main__":
    demo.launch()

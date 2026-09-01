import gradio as gr
import cv2
import numpy as np
from PIL import Image
import pytesseract


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



def classify_component(text):

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


def sketch_to_html(image):

    img = np.array(image)
    display_img = img.copy()

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY
    )[1]

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ==========================
    # OCR DETECTION
    # ==========================

    data = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DICT
    )

    components = []

    print("\n===== OCR WORDS =====")

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if not text:
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        print(
            f"{text} -> x={x}, y={y}, w={w}, h={h}"
        )

        components.append(
            {
                "text": text,
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }
        )

        cv2.rectangle(
            display_img,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    components = sorted(
        components,
        key=lambda c: (c["y"], c["x"])
    )

    full_text = " ".join(
        [c["text"] for c in components]
    )

    print("\n===== OCR TEXT =====")
    print(full_text)

    # ==========================
    # OCR CLASSIFICATION
    # ==========================

    ui_components = []

    for comp in components:

        component_type = classify_component(
            comp["text"]
        )

        ui_components.append(
            {
                "type": component_type,
                "text": comp["text"],
                "x": comp["x"],
                "y": comp["y"]
            }
        )

    # ==========================
    # SHAPE DETECTION
    # ==========================

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = w * h

        if (
            area > 500
            and w > 10
            and h > 10
        ):

            candidate = (
                x,
                y,
                w,
                h
            )

            if not is_duplicate(
                candidate,
                boxes
            ):

                boxes.append(candidate)

                shape_type = classify_shape(
                    w,
                    h
                )

                ui_components.append(
                    {
                        "type": shape_type,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h
                    }
                )

    sorted_boxes = sorted(
        boxes,
        key=lambda b: (b[1], b[0])
    )

    print("\n===== BOXES =====")

    for i, (x, y, w, h) in enumerate(
        sorted_boxes
    ):

        print(
            f"Box {i}: {(x, y, w, h)}"
        )

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

    # ==========================
    # HTML GENERATION
    # ==========================

    html_code = """
<div style="
max-width:400px;
margin:auto;
padding:20px;
font-family:Arial;
">
"""

    for comp in ui_components:

        if comp["type"] == "text":

            html_code += f"""
<p>{comp['text']}</p>
"""

        elif comp["type"] == "email":

            html_code += """
<input
type="email"
placeholder="Email"
style="
width:100%;
padding:12px;
margin-bottom:15px;
">
"""

        elif comp["type"] == "password":

            html_code += """
<input
type="password"
placeholder="Password"
style="
width:100%;
padding:12px;
margin-bottom:15px;
">
"""

        elif comp["type"] == "input":

            html_code += """
<input
type="text"
placeholder="Input"
style="
width:100%;
padding:12px;
margin-bottom:15px;
">
"""

        elif comp["type"] == "button":

            html_code += f"""
<button
style="
width:100%;
padding:12px;
margin-bottom:15px;
">
{comp.get('text', 'Button')}
</button>
"""

        elif comp["type"] == "navbar":

            html_code += """
<nav style="
background:#1976d2;
color:white;
padding:15px;
margin-bottom:20px;
">
Navbar
</nav>
"""

        elif comp["type"] == "card":

            html_code += """
<div style="
border:1px solid #ccc;
padding:20px;
margin-bottom:15px;
border-radius:8px;
">
Card Component
</div>
"""

    html_code += "</div>"

    # ==========================
    # REACT
    # ==========================

    react_code = """
export default function Website() {
  return (
    <div>
      <nav className="bg-blue-600 text-white p-4 flex justify-between">
        <span>Mobile Version</span>

        <div className="space-x-4">
          <a href="#">Home</a>
          <a href="#">Features</a>
          <a href="#">Contact</a>
        </div>
      </nav>

      <section className="bg-gray-100 text-center p-12">
        <h1 className="text-4xl font-bold">
          Hero Banner
        </h1>

        <p className="mt-4">
          Multiple Slide Images
        </p>
      </section>

      <section className="p-10">
        <h2 className="text-2xl font-bold">
          About Us
        </h2>

        <div className="border rounded p-6 mt-4">
          Company Image
        </div>
      </section>

      <section className="p-10">
        <h2 className="text-2xl font-bold">
          Sponsors
        </h2>

        <div className="border rounded p-6 mt-4">
          Sponsor Logos
        </div>
      </section>

      <section className="p-10">
        <h2 className="text-2xl font-bold">
          Features
        </h2>

        <div className="border rounded p-6 mt-4">
          Feature 1
        </div>

        <div className="border rounded p-6 mt-4">
          Feature 2
        </div>

        <div className="border rounded p-6 mt-4">
          Feature 3
        </div>
      </section>

      <section className="p-10">
        <h2 className="text-2xl font-bold">
          Contact
        </h2>

        <input
          className="border p-2 w-full mb-4"
          placeholder="Phone"
        />

        <input
          className="border p-2 w-full mb-4"
          placeholder="Email"
        />

        <button
          className="bg-blue-600 text-white px-4 py-2 rounded"
        >
          Submit
        </button>
      </section>

      <footer className="bg-black text-white p-6">
        Copyright | Sitemap | Contact Info
      </footer>
    </div>
  );
}
"""

    # ==========================
    # TAILWIND
    # ==========================

    tailwind_code = """
<div class="max-w-md mx-auto p-6">
  <input
    class="w-full border p-2 mb-4"
    placeholder="Email"
  />

  <input
    class="w-full border p-2 mb-4"
    placeholder="Password"
  />

  <button
    class="bg-blue-600 text-white px-4 py-2"
  >
    Login
  </button>
</div>
"""

    detected_image = Image.fromarray(
        display_img
    )

    summary_html = f"""
<h2>SketchNet Analysis</h2>

<p>
<strong>Detected Components:</strong>
{len(sorted_boxes)}
</p>

<h3>OCR Text</h3>

<pre>{full_text}</pre>
"""

    return (
        html_code,
        react_code,
        tailwind_code,
        summary_html + html_code,
        detected_image
    )


demo = gr.Interface(
    fn=sketch_to_html,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Code(
            language="html",
            label="Generated HTML"
        ),
        gr.Code(
            language="javascript",
            label="Generated React"
        ),
        gr.Code(
            language="html",
            label="Tailwind Code"
        ),
        gr.HTML(
            label="Preview"
        ),
        gr.Image(
            label="Detected Components"
        )
    ],
    title="SketchNet v1.0"
)

demo.launch()
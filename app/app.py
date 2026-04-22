import gradio as gr
import torch
import numpy as np
from PIL import Image
import json

from models.classifier import load_classifier, run_classifier
from models.detector import load_detector, run_detector
from models.segmentor import load_segmentor, run_segmentor
from models.sod import load_sod, run_sod

# ── Load all models once at startup ─────────────────────────────────────────
print("Loading models...")
classifier  = load_classifier("weights/best_model.pth")
detector    = load_detector("weights/detection_best.pt")
segmentor   = load_segmentor("weights/segmentation_best.pt")
sod_model   = load_sod("weights/best_u2net.pth")
print("All models loaded.")

CLASSES = ["Dent", "Scratch", "Crack", "Glass Shatter", "Lamp Broken", "Tire Flat"]

BEST_THRESHOLDS = {
    "Dent":          0.45,
    "Scratch":       0.45,
    "Crack":         0.40,
    "Glass Shatter": 0.50,
    "Lamp Broken":   0.45,
    "Tire Flat":     0.35,
}

CLASS_COLORS = {
    "Dent":          "#3B82F6",
    "Scratch":       "#F59E0B",
    "Crack":         "#EF4444",
    "Glass Shatter": "#8B5CF6",
    "Lamp Broken":   "#F97316",
    "Tire Flat":     "#10B981",
}


def analyze(image: Image.Image):
    if image is None:
        return None, None, None, "<p>Please upload an image.</p>"

    # ── Task 1: Classification ───────────────────────────────────────────────
    probs = run_classifier(classifier, image)           # dict {class: prob}
    detected = {c: p for c, p in probs.items() if p >= BEST_THRESHOLDS[c]}

    # Build classification HTML
    cls_html = build_classification_html(probs, detected)

    # ── Task 2: Detection ────────────────────────────────────────────────────
    det_image = run_detector(detector, image)           # PIL image with boxes

    # ── Task 3: Segmentation ─────────────────────────────────────────────────
    seg_image = run_segmentor(segmentor, image)         # PIL image with masks

    # ── Task 4: SOD ──────────────────────────────────────────────────────────
    sod_image = run_sod(sod_model, image)               # PIL saliency map

    return det_image, seg_image, sod_image, cls_html


def build_classification_html(probs: dict, detected: dict) -> str:
    rows = ""
    for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
        pct     = prob * 100
        color   = CLASS_COLORS[cls]
        active  = cls in detected
        badge   = f'<span class="badge" style="background:{color}">Detected</span>' if active else ""
        rows += f"""
        <div class="cls-row {'active' if active else ''}">
            <div class="cls-label">
                <span class="dot" style="background:{color}"></span>
                {cls} {badge}
            </div>
            <div class="bar-wrap">
                <div class="bar" style="width:{pct:.1f}%;background:{color}"></div>
            </div>
            <div class="pct">{pct:.1f}%</div>
        </div>"""

    summary = ", ".join(detected.keys()) if detected else "No damage detected"
    n = len(detected)
    header_color = "#EF4444" if n > 0 else "#10B981"

    return f"""
    <style>
      .cls-wrap {{ font-family: 'DM Sans', sans-serif; padding: 8px 0; }}
      .summary {{ font-size:1.1em; font-weight:700; color:{header_color};
                  margin-bottom:16px; padding:10px 14px;
                  background:{'#FEF2F2' if n>0 else '#F0FDF4'};
                  border-radius:8px; border-left:4px solid {header_color}; }}
      .cls-row {{ display:flex; align-items:center; gap:10px;
                  margin-bottom:10px; padding:8px 10px; border-radius:8px;
                  background:#F9FAFB; transition:background 0.2s; }}
      .cls-row.active {{ background:#FFF7ED; }}
      .cls-label {{ width:160px; font-size:.9em; font-weight:600;
                    color:#374151; display:flex; align-items:center; gap:6px; flex-wrap:wrap; }}
      .dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
      .badge {{ font-size:.7em; padding:2px 7px; border-radius:20px;
                color:white; font-weight:700; }}
      .bar-wrap {{ flex:1; background:#E5E7EB; border-radius:20px; height:10px; }}
      .bar {{ height:10px; border-radius:20px; transition:width 0.6s ease; }}
      .pct {{ width:48px; text-align:right; font-size:.85em;
              font-weight:700; color:#6B7280; }}
    </style>
    <div class="cls-wrap">
      <div class="summary">🔍 {n} damage type{'s' if n!=1 else ''} found: {summary}</div>
      {rows}
    </div>"""


# ── UI ───────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Syne:wght@700;800&display=swap');

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: #0F0F0F !important;
}

h1 { display: none; }

.app-header {
    text-align: center;
    padding: 36px 20px 24px;
}
.app-header .title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6em;
    font-weight: 800;
    color: #F9FAFB;
    letter-spacing: -1px;
    line-height: 1.1;
}
.app-header .title span { color: #F97316; }
.app-header .sub {
    color: #9CA3AF;
    font-size: 1em;
    margin-top: 8px;
}

.gr-panel, .gr-box, .svelte-1gfkn6j {
    background: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 12px !important;
}

.gr-button-primary {
    background: #F97316 !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1em !important;
    padding: 12px 28px !important;
    border-radius: 8px !important;
    color: white !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.gr-button-primary:hover { opacity: 0.85 !important; }

.gr-button-secondary {
    background: #2A2A2A !important;
    border: 1px solid #3A3A3A !important;
    color: #9CA3AF !important;
    border-radius: 8px !important;
}

label.svelte-1b6s6s, .gr-input-label {
    color: #9CA3AF !important;
    font-size: .85em !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: .05em;
}

.tabs { background: transparent !important; }
.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #6B7280 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.tab-nav button.selected {
    color: #F97316 !important;
    border-bottom-color: #F97316 !important;
}

.footer-note {
    text-align: center;
    color: #4B5563;
    font-size: .8em;
    padding: 16px;
}
"""

with gr.Blocks(css=css, title="Car Damage Detector") as demo:

    gr.HTML("""
    <div class="app-header">
        <div class="title">Car Damage <span>Detector</span></div>
        <div class="sub">Upload a photo — 4 AI models analyze it simultaneously</div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Car Image")
            analyze_btn = gr.Button("🔍 Analyze Damage", variant="primary")
            clear_btn   = gr.ClearButton(label="Clear")

            gr.HTML("""
            <div style="margin-top:16px; padding:14px; background:#1A1A1A;
                        border:1px solid #2A2A2A; border-radius:10px; color:#6B7280; font-size:.85em;">
                <b style="color:#9CA3AF">4 Models Running:</b><br><br>
                🏷️ &nbsp;<b style="color:#F97316">Classification</b> — EfficientNet-B3<br>
                📦 &nbsp;<b style="color:#3B82F6">Detection</b> — YOLOv8m<br>
                🎭 &nbsp;<b style="color:#8B5CF6">Segmentation</b> — YOLOv8m-seg<br>
                🌊 &nbsp;<b style="color:#10B981">Saliency</b> — U2-Net
            </div>
            """)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("🏷️ Classification"):
                    cls_output = gr.HTML(label="Classification Results")

                with gr.Tab("📦 Detection"):
                    det_output = gr.Image(label="Bounding Boxes", type="pil")

                with gr.Tab("🎭 Segmentation"):
                    seg_output = gr.Image(label="Instance Masks", type="pil")

                with gr.Tab("🌊 Saliency Map"):
                    sod_output = gr.Image(label="Salient Region", type="pil")

    analyze_btn.click(
        fn=analyze,
        inputs=[image_input],
        outputs=[det_output, seg_output, sod_output, cls_output]
    )
    clear_btn.add([image_input, det_output, seg_output, sod_output, cls_output])

    gr.HTML('<div class="footer-note">CarDD Dataset · EfficientNet-B3 · YOLOv8m · YOLOv8m-seg · U2-Net</div>')

if __name__ == "__main__":
    demo.launch()

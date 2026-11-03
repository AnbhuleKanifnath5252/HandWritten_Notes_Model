"""
High-Accuracy Merged Handwritten Notes & Scanned PDF Studio
Powered by PaddleOCR-VL + Qwen2.5-VL
Features:
 - Non-blocking background worker with live streaming (TextIteratorStreamer)
 - High-accuracy pre-processing (Auto-contrast & sharpness enhancement for faint notes)
 - Dual-model cross-validation pipeline (PaddleOCR character detection + Qwen visual reasoning)
 - Scanned PDF support (multi-page batching, page preview, high DPI rendering)
 - Low-RAM safeguards (garbage collection & sequential execution)
 - Real-time Stop / Cancel button
"""

import os
import sys
import gc
import io
import time
import tempfile
import threading
from threading import Thread
import torch
from PIL import Image, ImageOps, ImageEnhance
import pymupdf
import gradio as gr
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    TextIteratorStreamer
)

# Ensure local qwen-vl-utils is found
current_dir = os.path.dirname(os.path.abspath(__file__))
qwen_utils_path = os.path.join(current_dir, "qwen-vl-utils", "src")
if os.path.exists(qwen_utils_path):
    sys.path.insert(0, qwen_utils_path)

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_UTILS = True
except ImportError:
    HAS_QWEN_UTILS = False

# Global state
loaded_models = {
    "paddle": {"model": None, "processor": None, "path": ""},
    "qwen": {"model": None, "processor": None, "path": ""}
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

DEFAULT_PADDLE_PATH = os.path.abspath("./models/PaddleOCR-VL") if os.path.exists("./models/PaddleOCR-VL") else "PaddlePaddle/PaddleOCR-VL"
DEFAULT_QWEN_PATH = os.path.abspath("./models/Qwen2.5-VL-7B") if os.path.exists("./models/Qwen2.5-VL-7B") else "Qwen/Qwen2.5-VL-7B-Instruct"


# =====================================================================
# High-Accuracy Image Preprocessing
# =====================================================================
def enhance_handwritten_image(image: Image.Image, auto_contrast: bool = True, sharpness_factor: float = 1.3) -> Image.Image:
    """Enhance contrast and sharpness so faint pencil/ink stands out against paper."""
    if image is None:
        return None
    try:
        img = image.convert("RGB")
        if auto_contrast:
            img = ImageOps.autocontrast(img, cutoff=1)
        if sharpness_factor > 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpness_factor)
        return img
    except Exception as e:
        print(f"Enhancement warning: {e}")
        return image


# =====================================================================
# PDF Processing (PyMuPDF)
# =====================================================================
def get_pdf_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, (list, tuple)):
        file_obj = file_obj[0] if len(file_obj) > 0 else None
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    if hasattr(file_obj, "path"):
        return file_obj.path
    if hasattr(file_obj, "name"):
        return file_obj.name
    return None


def get_pdf_page_count(pdf_path):
    if not pdf_path or not os.path.exists(pdf_path):
        return 0
    try:
        doc = pymupdf.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return 0


def render_pdf_page(pdf_path, page_number=0, dpi=150, enhance=True):
    if not pdf_path or not os.path.exists(pdf_path):
        return None
    try:
        doc = pymupdf.open(pdf_path)
        if page_number < 0 or page_number >= len(doc):
            page_number = 0
        page = doc[page_number]
        pix = page.get_pixmap(dpi=int(dpi or 150))
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()

        # Prevent browser payload/memory crashes on giant scans
        max_dim = 1800
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        if enhance:
            img = enhance_handwritten_image(img)
        return img
    except Exception as e:
        print(f"Error rendering PDF page {page_number}: {e}")
        return None


# =====================================================================
# Memory Management & Model Loading
# =====================================================================
def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_paddle_model(model_path):
    if loaded_models["paddle"]["model"] is not None and loaded_models["paddle"]["path"] == model_path:
        return "[PaddleOCR-VL] Already loaded."

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        model.eval()

        loaded_models["paddle"]["model"] = model
        loaded_models["paddle"]["processor"] = processor
        loaded_models["paddle"]["path"] = model_path
        return f"[PaddleOCR-VL] Loaded from {model_path} ({DEVICE})"
    except Exception as e:
        return f"[PaddleOCR-VL] Load Error: {str(e)}"


def load_qwen_model(model_path):
    if loaded_models["qwen"]["model"] is not None and loaded_models["qwen"]["path"] == model_path:
        return "[Qwen2.5-VL] Already loaded."

    try:
        processor = AutoProcessor.from_pretrained(model_path)
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        model.eval()

        loaded_models["qwen"]["model"] = model
        loaded_models["qwen"]["processor"] = processor
        loaded_models["qwen"]["path"] = model_path
        return f"[Qwen2.5-VL] Loaded from {model_path} ({DEVICE})"
    except Exception as e:
        return f"[Qwen2.5-VL] Load Error: {str(e)}"


def unload_paddle_model():
    """Unload Paddle to save RAM before heavy Qwen reasoning."""
    if loaded_models["paddle"]["model"] is not None:
        del loaded_models["paddle"]["model"]
        loaded_models["paddle"]["model"] = None
        free_memory()


# =====================================================================
# High-Accuracy Prompts
# =====================================================================
HIGH_ACCURACY_PROMPT_TEMPLATE = """You are a world-class handwriting transcription specialist and document preservation expert.

PRIMARY INPUTS:
1. Visual scan of the handwritten note/document.
2. Raw character & formula extract detected by the optical recognition model (PaddleOCR-VL):
--- RAW OCR EXTRACT ---
{paddle_text}
--- END RAW OCR EXTRACT ---

USER SPECIFIC INSTRUCTIONS:
{user_prompt}

STRICT HIGH-ACCURACY RULES:
1. RIGOROUS CROSS-VALIDATION:
   - Carefully verify every word in the visual scan against the raw OCR extract.
   - Correct optical misrecognitions (e.g. misread cursive, numbers vs letters like 0 vs O, 1 vs l, 5 vs S).
   - If a word is partially ambiguous, use the semantic and scientific context of the notes to resolve it accurately.
   - NEVER hallucinate, extrapolate, or invent information not present in the original document.

2. MATHEMATICS, FORMULAS & NOTATION:
   - Convert all math expressions, formulas, and chemical symbols into standard LaTeX.
   - Use $...$ for inline math (e.g., $E = mc^2$, $\\int_0^\\infty f(x)dx$).
   - Use $$...$$ on its own line for display equations.
   - Preserve superscripts, subscripts, fractions (\\frac{{a}}{{b}}), matrices, and Greek letters (\\alpha, \\beta, \\theta).

3. DOCUMENT STRUCTURE:
   - Faithfully preserve headings, bulleted lists, numbered steps, and tables.
   - Format tables in standard Markdown table syntax.
   - If there are margin notes, boxed rules, or important callouts, format them as blockquotes (> ...).
   - Disregard paper artifacts (binder holes, coffee marks, scanner borders).

Produce the final, polished, highly accurate transcription below in clean Markdown:"""


# =====================================================================
# Background Inference Workers (Streaming)
# =====================================================================
def run_paddle_ocr(image, prompt="OCR:", max_tokens=1024):
    if loaded_models["paddle"]["model"] is None:
        return "PaddleOCR-VL model is not loaded."

    model = loaded_models["paddle"]["model"]
    processor = loaded_models["paddle"]["processor"]

    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.05
            )

        result = processor.decode(output_ids[0], skip_special_tokens=True)
        if result.startswith(prompt):
            result = result[len(prompt):].strip()
        return result
    except Exception as e:
        return f"PaddleOCR-VL error: {str(e)}"


def stream_qwen_vl(image, prompt, max_tokens=2048):
    """Streams Qwen2.5-VL tokens in real-time using TextIteratorStreamer."""
    if loaded_models["qwen"]["model"] is None:
        yield "Qwen2.5-VL model is not loaded."
        return

    model = loaded_models["qwen"]["model"]
    processor = loaded_models["qwen"]["processor"]

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if HAS_QWEN_UTILS:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Background streaming generation
        streamer = TextIteratorStreamer(processor.tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "streamer": streamer,
            "do_sample": False,
            "repetition_penalty": 1.05
        }

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated_text = ""
        for chunk in streamer:
            accumulated_text += chunk
            yield accumulated_text

        thread.join()
    except Exception as e:
        yield f"Qwen2.5-VL error: {str(e)}"


# =====================================================================
# Main Multi-Stage Processing Pipeline (Generator / Background)
# =====================================================================
def execute_pipeline(
    input_type,
    pdf_file,
    pdf_page_num,
    pdf_scope,
    pdf_dpi,
    direct_image,
    auto_enhance,
    mode,
    custom_prompt,
    max_tokens,
    low_ram_mode,
    paddle_path,
    qwen_path
):
    """
    Generator function that runs in the background and streams live updates to the UI.
    """
    # 1. Determine targets
    images_to_process = []
    page_labels = []

    if input_type == "📄 Scanned PDF":
        pdf_path = get_pdf_path(pdf_file)
        if not pdf_path:
            yield "⚠️ Please upload a valid PDF file first.", "", "", "Idle"
            return

        total_pages = get_pdf_page_count(pdf_path)
        if total_pages == 0:
            yield "⚠️ Could not read any pages from the uploaded PDF.", "", "", "Idle"
            return

        if pdf_scope == "Current Selected Page":
            target_idx = int(pdf_page_num) - 1
            yield "", "", "", f"⏳ [1/1] Rendering Page {pdf_page_num} of {total_pages}..."
            img = render_pdf_page(pdf_path, target_idx, dpi=int(pdf_dpi), enhance=auto_enhance)
            images_to_process.append(img)
            page_labels.append(f"Page {pdf_page_num} of {total_pages}")
        else:
            # All pages
            yield "", "", "", f"⏳ Rendering {total_pages} pages from PDF..."
            for p in range(total_pages):
                img = render_pdf_page(pdf_path, p, dpi=int(pdf_dpi), enhance=auto_enhance)
                images_to_process.append(img)
                page_labels.append(f"Page {p + 1} of {total_pages}")
    else:
        if direct_image is None:
            yield "⚠️ Please upload an image or take a photo first.", "", "", "Idle"
            return
        img = enhance_handwritten_image(direct_image) if auto_enhance else direct_image
        images_to_process.append(img)
        page_labels.append("Handwritten Note Image")

    # 2. Check and load models
    needs_paddle = mode in ["Merged Pipeline (Both Models - Highly Accurate)", "PaddleOCR-VL Only", "Side-by-Side Comparison"]
    needs_qwen = mode in ["Merged Pipeline (Both Models - Highly Accurate)", "Qwen2.5-VL Only", "Side-by-Side Comparison"]

    if needs_paddle and loaded_models["paddle"]["model"] is None:
        yield "", "", "", "⏳ [Background] Pre-loading PaddleOCR-VL model weights..."
        res = load_paddle_model(paddle_path)
        if "Error" in res:
            yield f"⚠️ {res}", "", "", "Error loading PaddleOCR-VL"
            return

    if needs_qwen and not low_ram_mode and loaded_models["qwen"]["model"] is None:
        yield "", "", "", "⏳ [Background] Pre-loading Qwen2.5-VL model weights..."
        res = load_qwen_model(qwen_path)
        if "Error" in res:
            yield f"⚠️ {res}", "", "", "Error loading Qwen2.5-VL"
            return

    # 3. Process pages
    total_items = len(images_to_process)
    compiled_final = ""
    compiled_paddle = ""
    compiled_qwen = ""

    for idx, (img, label) in enumerate(zip(images_to_process, page_labels)):
        item_prefix = f"[{idx + 1}/{total_items}] {label}"

        # Stage A: PaddleOCR-VL
        paddle_out = ""
        if needs_paddle:
            yield (
                compiled_final,
                compiled_paddle,
                compiled_qwen,
                f"⚙️ {item_prefix} - Step 1/2: Running PaddleOCR-VL character & formula detection..."
            )
            paddle_out = run_paddle_ocr(img, prompt="OCR:", max_tokens=max_tokens)
            compiled_paddle += f"### {label}\n{paddle_out}\n\n"

        # Sequential low-RAM offload
        if low_ram_mode and needs_paddle and needs_qwen:
            yield compiled_final, compiled_paddle, compiled_qwen, f"🧹 Offloading PaddleOCR to free RAM before Qwen reasoning..."
            unload_paddle_model()

        # Ensure Qwen loaded
        if needs_qwen and loaded_models["qwen"]["model"] is None:
            yield compiled_final, compiled_paddle, compiled_qwen, f"⏳ [Background] Loading Qwen2.5-VL weights..."
            load_qwen_model(qwen_path)

        # Stage B: Qwen2.5-VL
        if mode == "Merged Pipeline (Both Models - Highly Accurate)":
            full_verification_prompt = HIGH_ACCURACY_PROMPT_TEMPLATE.format(
                paddle_text=paddle_out,
                user_prompt=custom_prompt
            )

            status_text = f"🧠 {item_prefix} - Step 2/2: Cross-referencing visual & OCR with Qwen2.5-VL..."
            current_page_qwen = ""

            for chunk in stream_qwen_vl(img, prompt=full_verification_prompt, max_tokens=max_tokens):
                current_page_qwen = chunk
                preview_combined = compiled_final + f"\n\n## 📄 {label}\n\n" + current_page_qwen
                yield preview_combined, compiled_paddle, compiled_qwen + f"### {label}\n{current_page_qwen}\n\n", status_text

            compiled_final += f"\n\n## 📄 {label}\n\n{current_page_qwen}"
            compiled_qwen += f"### {label}\n{current_page_qwen}\n\n"

        elif mode == "Side-by-Side Comparison":
            status_text = f"🧠 {item_prefix} - Running Qwen2.5-VL independently..."
            current_page_qwen = ""
            for chunk in stream_qwen_vl(img, prompt=custom_prompt, max_tokens=max_tokens):
                current_page_qwen = chunk
                preview_combined = (
                    compiled_final +
                    f"\n\n## 📄 {label}\n\n"
                    f"**🟢 PaddleOCR-VL:**\n\n{paddle_out}\n\n"
                    f"---\n\n**🔵 Qwen2.5-VL:**\n\n{current_page_qwen}"
                )
                yield preview_combined, compiled_paddle, compiled_qwen + f"### {label}\n{current_page_qwen}\n\n", status_text

            compiled_final += (
                f"\n\n## 📄 {label}\n\n"
                f"**🟢 PaddleOCR-VL:**\n\n{paddle_out}\n\n"
                f"---\n\n**🔵 Qwen2.5-VL:**\n\n{current_page_qwen}"
            )
            compiled_qwen += f"### {label}\n{current_page_qwen}\n\n"

        elif mode == "PaddleOCR-VL Only":
            compiled_final += f"\n\n## 📄 {label}\n\n{paddle_out}"
            yield compiled_final, compiled_paddle, "N/A", f"✅ {item_prefix} Complete."

        elif mode == "Qwen2.5-VL Only":
            status_text = f"🧠 {item_prefix} - Running Qwen2.5-VL..."
            current_page_qwen = ""
            for chunk in stream_qwen_vl(img, prompt=custom_prompt, max_tokens=max_tokens):
                current_page_qwen = chunk
                preview_combined = compiled_final + f"\n\n## 📄 {label}\n\n{current_page_qwen}"
                yield preview_combined, "N/A", compiled_qwen + f"### {label}\n{current_page_qwen}\n\n", status_text

            compiled_final += f"\n\n## 📄 {label}\n\n{current_page_qwen}"
            compiled_qwen += f"### {label}\n{current_page_qwen}\n\n"

        free_memory()

    yield compiled_final.strip(), compiled_paddle.strip(), compiled_qwen.strip(), f"🎉 All {total_items} item(s) processed successfully with maximum accuracy!"


def export_markdown_file(markdown_text):
    if not markdown_text or not markdown_text.strip():
        return None
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, "transcribed_notes_accurate.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    return filepath


# =====================================================================
# Gradio Modern Frontend
# =====================================================================
CUSTOM_CSS = """
#main-title { text-align: center; margin-bottom: 4px; font-weight: 700; }
#subtitle { text-align: center; color: #4b5563; margin-bottom: 18px; font-size: 1.05em; }
.status-box { font-weight: 600; padding: 8px 12px; border-radius: 6px; background: #f3f4f6; }
"""

def create_ui():

    with gr.Blocks(title="Handwritten Notes Studio: PaddleOCR-VL + Qwen2.5-VL") as demo:
        gr.Markdown("# ✍️ High-Accuracy Handwritten Notes Studio", elem_id="main-title")
        gr.Markdown(
            "Dual-Model Background Engine: **PaddleOCR-VL** (Optical Detection) + **Qwen2.5-VL** (Visual Reasoning) | **Scanned PDFs & Images**",
            elem_id="subtitle"
        )

        with gr.Row():
            # ================= Left Column: Input and Controls =================
            with gr.Column(scale=1):
                input_type = gr.Radio(
                    choices=["📄 Scanned PDF", "🖼️ Direct Image / Photo"],
                    value="📄 Scanned PDF",
                    label="Input Source"
                )

                # PDF Group
                with gr.Group(visible=True) as pdf_group:
                    pdf_file = gr.File(label="Upload Scanned PDF File (.pdf)", file_types=[".pdf"])
                    with gr.Row():
                        pdf_status = gr.Markdown("No PDF loaded")
                        pdf_scope = gr.Radio(
                            choices=["Current Selected Page", "All Pages (Entire PDF)"],
                            value="Current Selected Page",
                            label="Processing Scope"
                        )
                    with gr.Row():
                        page_slider = gr.Slider(
                            minimum=1, maximum=10, step=1, value=1, label="Page Selector", interactive=False
                        )
                        pdf_dpi = gr.Slider(
                            minimum=100, maximum=300, step=25, value=175, label="Rendering DPI (Quality)"
                        )
                    with gr.Row():
                        prev_page_btn = gr.Button("◀ Prev Page", size="sm")
                        next_page_btn = gr.Button("Next Page ▶", size="sm")

                    pdf_preview = gr.Image(type="pil", label="Scanned Page Preview", interactive=False)

                # Direct Image Group
                with gr.Group(visible=False) as image_group:
                    direct_image = gr.Image(type="pil", label="Upload Handwritten Note / Photo")

                mode_select = gr.Radio(
                    choices=[
                        "Merged Pipeline (Both Models - Highly Accurate)",
                        "Side-by-Side Comparison",
                        "PaddleOCR-VL Only",
                        "Qwen2.5-VL Only"
                    ],
                    value="Merged Pipeline (Both Models - Highly Accurate)",
                    label="Execution Mode"
                )

                with gr.Accordion("🎯 Accuracy & Image Optimization", open=True):
                    auto_enhance = gr.Checkbox(
                        label="Auto-Enhance Contrast & Sharpness (Recommended for faint pencil/ink notes)",
                        value=True
                    )
                    prompt_templates = gr.Dropdown(
                        label="Task Preset",
                        choices=[
                            "Transcribe the handwritten notes into structured Markdown and LaTeX formulas.",
                            "Transcribe every word verbatim and preserve the exact original layout.",
                            "Explain the handwritten solution or lecture notes step-by-step.",
                            "Extract all math equations, definitions, and theorems from the notes.",
                            "Custom"
                        ],
                        value="Transcribe the handwritten notes into structured Markdown and LaTeX formulas."
                    )
                    custom_prompt_input = gr.Textbox(
                        label="Custom Instruction",
                        value="Transcribe the handwritten notes into structured Markdown and LaTeX formulas.",
                        lines=2
                    )

                with gr.Accordion("⚙️ Hardware & Model Paths", open=False):
                    low_ram_mode = gr.Checkbox(
                        label="Low-RAM Mode (Offloads PaddleOCR before running Qwen - Safe for 8GB RAM)",
                        value=True
                    )
                    paddle_path_input = gr.Textbox(
                        label="PaddleOCR-VL Path or Repo ID",
                        value=DEFAULT_PADDLE_PATH
                    )
                    qwen_path_input = gr.Textbox(
                        label="Qwen2.5-VL Path or Repo ID",
                        value=DEFAULT_QWEN_PATH,
                        info="For 8GB RAM on CPU, recommend 'Qwen/Qwen2.5-VL-3B-Instruct'."
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=256, maximum=3072, value=1536, step=128, label="Max New Tokens"
                    )

                with gr.Row():
                    run_btn = gr.Button("🚀 Run Analysis (Background)", variant="primary", size="lg", scale=3)
                    stop_btn = gr.Button("⏹ Stop", variant="stop", size="lg", scale=1)

                live_status = gr.Markdown("**Status:** Idle", elem_classes=["status-box"])

            # ================= Right Column: Results & Export =================
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("🌟 Final Verified Markdown Result"):
                        final_output = gr.Markdown(label="Final Transcription")
                        with gr.Row():
                            download_btn = gr.Button("💾 Export as Markdown (.md)", size="sm")
                            download_file = gr.File(label="Download File", visible=False)

                    with gr.TabItem("📊 Multi-Model Raw Extracts"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### 🟢 PaddleOCR-VL (Optical Characters)")
                                paddle_raw_output = gr.Textbox(label="PaddleOCR Output", lines=20)
                            with gr.Column():
                                gr.Markdown("#### 🔵 Qwen2.5-VL (Visual Reasoning)")
                                qwen_raw_output = gr.Textbox(label="Qwen2.5-VL Output", lines=20)

        # ================= UI Event Wiring =================
        def switch_source(choice):
            return gr.update(visible=(choice == "📄 Scanned PDF")), gr.update(visible=(choice != "📄 Scanned PDF"))

        input_type.change(
            fn=switch_source,
            inputs=[input_type],
            outputs=[pdf_group, image_group]
        )

        def on_pdf_uploaded(file, dpi, enhance):
            try:
                pdf_path = get_pdf_path(file)
                if not pdf_path or not os.path.exists(pdf_path):
                    return gr.update(minimum=1, maximum=10, value=1, interactive=False), "⚠️ No valid PDF received", None
                total = get_pdf_page_count(pdf_path)
                if total == 0:
                    return gr.update(minimum=1, maximum=10, value=1, interactive=False), "⚠️ Invalid or empty PDF", None
                preview = render_pdf_page(pdf_path, 0, dpi=int(dpi or 150), enhance=bool(enhance))
                slider_max = max(2, total)
                return (
                    gr.update(minimum=1, maximum=slider_max, value=1, interactive=(total > 1)),
                    f"✅ **Loaded PDF:** {total} page(s)",
                    preview
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                return gr.update(minimum=1, maximum=10, value=1, interactive=False), f"⚠️ Error loading PDF: {str(e)}", None

        pdf_file.upload(
            fn=on_pdf_uploaded,
            inputs=[pdf_file, pdf_dpi, auto_enhance],
            outputs=[page_slider, pdf_status, pdf_preview]
        )

        pdf_file.clear(
            fn=lambda: (gr.update(minimum=1, maximum=10, value=1, interactive=False), "No PDF loaded", None),
            outputs=[page_slider, pdf_status, pdf_preview]
        )

        def on_page_changed(file, page_num, dpi, enhance):
            try:
                pdf_path = get_pdf_path(file)
                if not pdf_path or not os.path.exists(pdf_path):
                    return None
                total = get_pdf_page_count(pdf_path)
                if total == 0:
                    return None
                p_idx = max(0, min(total - 1, int(page_num or 1) - 1))
                return render_pdf_page(pdf_path, p_idx, dpi=int(dpi or 150), enhance=bool(enhance))
            except Exception as e:
                print(f"Error in on_page_changed: {e}")
                return None

        page_slider.change(
            fn=on_page_changed,
            inputs=[pdf_file, page_slider, pdf_dpi, auto_enhance],
            outputs=[pdf_preview]
        )

        pdf_dpi.change(
            fn=on_page_changed,
            inputs=[pdf_file, page_slider, pdf_dpi, auto_enhance],
            outputs=[pdf_preview]
        )

        def navigate_page(file, current_page, direction):
            try:
                pdf_path = get_pdf_path(file)
                if not pdf_path or not os.path.exists(pdf_path):
                    return 1
                total = get_pdf_page_count(pdf_path)
                curr = int(current_page or 1)
                new_page = max(1, min(total, curr + direction))
                return new_page
            except Exception:
                return 1

        prev_page_btn.click(
            fn=lambda f, p: navigate_page(f, p, -1),
            inputs=[pdf_file, page_slider],
            outputs=[page_slider]
        )

        next_page_btn.click(
            fn=lambda f, p: navigate_page(f, p, 1),
            inputs=[pdf_file, page_slider],
            outputs=[page_slider]
        )

        prompt_templates.change(
            fn=lambda x, cur: x if x != "Custom" else cur,
            inputs=[prompt_templates, custom_prompt_input],
            outputs=[custom_prompt_input]
        )

        # Background processing with cancellation support
        run_event = run_btn.click(
            fn=execute_pipeline,
            inputs=[
                input_type,
                pdf_file,
                page_slider,
                pdf_scope,
                pdf_dpi,
                direct_image,
                auto_enhance,
                mode_select,
                custom_prompt_input,
                max_tokens_slider,
                low_ram_mode,
                paddle_path_input,
                qwen_path_input
            ],
            outputs=[
                final_output,
                paddle_raw_output,
                qwen_raw_output,
                live_status
            ]
        )

        stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[run_event]
        )

        def handle_export(text):
            out_path = export_markdown_file(text)
            if out_path:
                return gr.update(value=out_path, visible=True)
            return gr.update(visible=False)

        download_btn.click(
            fn=handle_export,
            inputs=[final_output],
            outputs=[download_file]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    # Launch with queue enabled so background streaming works seamlessly
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CUSTOM_CSS)

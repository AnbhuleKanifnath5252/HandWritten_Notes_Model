"""
Universal Handwritten Notes & Document Studio
Powered by PaddleOCR-VL + Qwen2.5-VL
Features:
 - Reads EVERY file format: Scanned PDF, Images (.png, .jpg, .webp, .tiff, .bmp), Word (.docx), Text (.txt, .md)
 - Automatically converts any input into a Standard Multi-Page PDF
 - High-accuracy dual-model pipeline (PaddleOCR-VL + Qwen2.5-VL)
 - Non-blocking streaming background execution with live token updates
 - Dual Export: Download as Markdown (.md) AND Standard Publication-Grade PDF (.pdf)
 - Low-RAM safeguard & on-the-fly model reload/unload
"""

import os
import sys

# Disable TensorFlow/OneDNN warnings & force pure PyTorch
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

# Optional document libraries
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

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
# Universal Document Converter -> Standard PDF
# =====================================================================
def get_file_path(file_obj):
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


def convert_any_file_to_standard_pdf(file_path):
    """
    Accepts ANY file format (PDF, PNG, JPG, WEBP, TIFF, BMP, DOCX, TXT, MD)
    and normalizes it into a standard PDF file path.
    """
    if not file_path or not os.path.exists(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()
    temp_dir = tempfile.gettempdir()
    out_pdf_path = os.path.join(temp_dir, f"normalized_{os.path.basename(file_path)}.pdf")

    # 1. Already a standard PDF
    if ext == ".pdf":
        return file_path

    # 2. Images (Single & Multi-page TIFF, PNG, JPG, BMP, WEBP)
    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"]:
        try:
            im = Image.open(file_path)
            frames = []
            try:
                while True:
                    frames.append(im.copy().convert("RGB"))
                    im.seek(len(frames))
            except EOFError:
                pass

            if frames:
                frames[0].save(out_pdf_path, "PDF", resolution=150.0, save_all=True, append_images=frames[1:])
            else:
                im.convert("RGB").save(out_pdf_path, "PDF", resolution=150.0)
            return out_pdf_path
        except Exception as e:
            print(f"Pillow image to PDF fallback to pymupdf: {e}")
            try:
                img_doc = pymupdf.open(file_path)
                pdf_bytes = img_doc.convert_to_pdf()
                with open(out_pdf_path, "wb") as f:
                    f.write(pdf_bytes)
                img_doc.close()
                return out_pdf_path
            except Exception as e2:
                print(f"Image conversion error: {e2}")
                return None

    # 3. Word Documents (.docx)
    if ext in [".docx", ".doc"] and HAS_DOCX:
        try:
            doc = docx.Document(file_path)
            styles = getSampleStyleSheet()
            h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=16, spaceAfter=8, textColor=colors.HexColor('#1e40af'))
            body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=4)

            story = []
            for para in doc.paragraphs:
                txt = para.text.strip()
                if not txt:
                    story.append(Spacer(1, 4))
                elif para.style.name.startswith("Heading"):
                    story.append(Paragraph(txt, h1))
                else:
                    clean = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    story.append(Paragraph(clean, body))

            pdf_doc = SimpleDocTemplate(out_pdf_path, pagesize=letter, margin=40)
            pdf_doc.build(story)
            return out_pdf_path
        except Exception as e:
            print(f"DOCX to PDF conversion error: {e}")

    # 4. Plain Text & Markdown (.txt, .md, .csv, .log)
    if ext in [".txt", ".md", ".csv", ".log"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return export_standard_pdf(content, out_pdf_path)
        except Exception as e:
            print(f"Text to PDF conversion error: {e}")

    # 5. Generic PyMuPDF fallback for other supported document formats (XPS, EPUB, etc.)
    try:
        generic_doc = pymupdf.open(file_path)
        pdf_bytes = generic_doc.convert_to_pdf()
        with open(out_pdf_path, "wb") as f:
            f.write(pdf_bytes)
        generic_doc.close()
        return out_pdf_path
    except Exception as e:
        print(f"Generic document conversion failed: {e}")
        return None


# =====================================================================
# PDF Page Extraction & Preprocessing (PyMuPDF + Pillow)
# =====================================================================
def get_pdf_page_count(pdf_path):
    if not pdf_path or not os.path.exists(pdf_path):
        return 0
    try:
        doc = pymupdf.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception as e:
        print(f"Error reading PDF page count: {e}")
        return 0


def enhance_handwritten_image(image: Image.Image, auto_contrast: bool = True, sharpness_factor: float = 1.3) -> Image.Image:
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

        # Prevent browser payload and memory crashes on giant scans
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
# Memory Management & Model Lifecycle
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
    if loaded_models["paddle"]["model"] is not None:
        del loaded_models["paddle"]["model"]
        loaded_models["paddle"]["model"] = None
        free_memory()


def unload_all_models():
    """Unloads all models from memory and frees RAM."""
    for key in ["paddle", "qwen"]:
        if loaded_models[key]["model"] is not None:
            del loaded_models[key]["model"]
            loaded_models[key]["model"] = None
        if loaded_models[key]["processor"] is not None:
            del loaded_models[key]["processor"]
            loaded_models[key]["processor"] = None
        loaded_models[key]["path"] = ""
    free_memory()
    return "🧹 All models unloaded. System RAM freed."


def reload_models(paddle_path, qwen_path):
    """Purges loaded models and re-initializes fresh weights from disk."""
    unload_all_models()
    res_p = load_paddle_model(paddle_path)
    res_q = load_qwen_model(qwen_path)
    return f"🔄 Reload Complete:\n- {res_p}\n- {res_q}"


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
   - If a word is partially ambiguous, use the semantic context of the notes to resolve it accurately.
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
# Model Inference Engines
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
# Main Multi-Stage Processing Pipeline (Background Generator)
# =====================================================================
def execute_pipeline(
    input_file,
    camera_image,
    page_num,
    scope,
    dpi,
    auto_enhance,
    mode,
    custom_prompt,
    max_tokens,
    low_ram_mode,
    paddle_path,
    qwen_path
):
    images_to_process = []
    page_labels = []

    # Check Camera Image First
    if camera_image is not None and input_file is None:
        img = enhance_handwritten_image(camera_image) if auto_enhance else camera_image
        images_to_process.append(img)
        page_labels.append("Captured Image")
    else:
        file_path = get_file_path(input_file)
        if not file_path:
            yield "⚠️ Please upload a document/image or capture a photo first.", "", "", "Idle"
            return

        yield "", "", "", "⏳ Converting & standardizing file to PDF..."
        pdf_path = convert_any_file_to_standard_pdf(file_path)
        if not pdf_path or not os.path.exists(pdf_path):
            yield "⚠️ Could not convert this file format to standard PDF.", "", "", "Idle"
            return

        total_pages = get_pdf_page_count(pdf_path)
        if total_pages == 0:
            yield "⚠️ Document has 0 readable pages.", "", "", "Idle"
            return

        if scope == "Current Selected Page":
            target_idx = max(0, min(total_pages - 1, int(page_num or 1) - 1))
            yield "", "", "", f"⏳ Rendering Page {target_idx + 1} of {total_pages}..."
            img = render_pdf_page(pdf_path, target_idx, dpi=int(dpi or 150), enhance=auto_enhance)
            if img is not None:
                images_to_process.append(img)
                page_labels.append(f"Page {target_idx + 1} of {total_pages}")
        else:
            yield "", "", "", f"⏳ Rendering all {total_pages} pages..."
            for p in range(total_pages):
                img = render_pdf_page(pdf_path, p, dpi=int(dpi or 150), enhance=auto_enhance)
                if img is not None:
                    images_to_process.append(img)
                    page_labels.append(f"Page {p + 1} of {total_pages}")

    if not images_to_process:
        yield "⚠️ No pages could be extracted from the document.", "", "", "Idle"
        return

    # Check and load models
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
                f"⚙️ {item_prefix} - Step 1/2: PaddleOCR-VL optical detection..."
            )
            paddle_out = run_paddle_ocr(img, prompt="OCR:", max_tokens=max_tokens)
            compiled_paddle += f"### {label}\n{paddle_out}\n\n"

        if low_ram_mode and needs_paddle and needs_qwen:
            yield compiled_final, compiled_paddle, compiled_qwen, "🧹 Offloading PaddleOCR to free RAM before Qwen reasoning..."
            unload_paddle_model()

        if needs_qwen and loaded_models["qwen"]["model"] is None:
            yield compiled_final, compiled_paddle, compiled_qwen, "⏳ [Background] Loading Qwen2.5-VL weights..."
            load_qwen_model(qwen_path)

        # Stage B: Qwen2.5-VL
        if mode == "Merged Pipeline (Both Models - Highly Accurate)":
            full_prompt = HIGH_ACCURACY_PROMPT_TEMPLATE.format(
                paddle_text=paddle_out,
                user_prompt=custom_prompt
            )
            status_text = f"🧠 {item_prefix} - Step 2/2: Reasoning & structuring with Qwen2.5-VL..."
            current_page_qwen = ""
            for chunk in stream_qwen_vl(img, prompt=full_prompt, max_tokens=max_tokens):
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
                    f"**🟢 PaddleOCR-VL:**\n\n{paddle_out}\n\n---\n\n**🔵 Qwen2.5-VL:**\n\n{current_page_qwen}"
                )
                yield preview_combined, compiled_paddle, compiled_qwen + f"### {label}\n{current_page_qwen}\n\n", status_text

            compiled_final += (
                f"\n\n## 📄 {label}\n\n"
                f"**🟢 PaddleOCR-VL:**\n\n{paddle_out}\n\n---\n\n**🔵 Qwen2.5-VL:**\n\n{current_page_qwen}"
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

    yield compiled_final.strip(), compiled_paddle.strip(), compiled_qwen.strip(), f"🎉 All {total_items} page(s) processed and formatted successfully!"


# =====================================================================
# Dual Exporter (Markdown & Standard PDF via ReportLab)
# =====================================================================
def export_markdown_file(markdown_text):
    if not markdown_text or not markdown_text.strip():
        return None
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, "transcribed_notes.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    return filepath


def export_standard_pdf(markdown_text, out_pdf_path=None):
    if not markdown_text or not markdown_text.strip():
        return None
    if out_pdf_path is None:
        temp_dir = tempfile.gettempdir()
        out_pdf_path = os.path.join(temp_dir, "transcribed_notes_publication.pdf")

    try:
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, leading=22, spaceAfter=10, textColor=colors.HexColor('#0f172a'))
        h1_style = ParagraphStyle('DocH1', parent=styles['Heading1'], fontSize=14, leading=18, spaceAfter=8, textColor=colors.HexColor('#1e40af'))
        h2_style = ParagraphStyle('DocH2', parent=styles['Heading2'], fontSize=12, leading=16, spaceAfter=6, textColor=colors.HexColor('#2563eb'))
        body_style = ParagraphStyle('DocBody', parent=styles['Normal'], fontSize=9.5, leading=13.5, spaceAfter=4, textColor=colors.HexColor('#334155'))

        story = []
        for line in markdown_text.split('\n'):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 4))
            elif line.startswith('# '):
                story.append(Paragraph(line[2:], title_style))
            elif line.startswith('## '):
                story.append(Paragraph(line[3:], h1_style))
            elif line.startswith('### '):
                story.append(Paragraph(line[4:], h2_style))
            elif line.startswith('---'):
                story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#cbd5e1'), spaceAfter=6))
            elif line.startswith('```'):
                continue
            else:
                clean = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(clean, body_style))

        pdf_doc = SimpleDocTemplate(
            out_pdf_path,
            pagesize=letter,
            rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45
        )
        pdf_doc.build(story)
        return out_pdf_path
    except Exception as e:
        print(f"Error generating standard PDF: {e}")
        return None


# =====================================================================
# Gradio Modern Frontend
# =====================================================================
CUSTOM_CSS = """
#main-title { text-align: center; margin-bottom: 2px; font-weight: 700; }
#subtitle { text-align: center; color: #4b5563; margin-bottom: 16px; font-size: 1.05em; }
.status-box { font-weight: 600; padding: 8px 12px; border-radius: 6px; background: #f3f4f6; }
"""

def create_ui():
    with gr.Blocks(title="Universal Notes & Document Studio: PaddleOCR-VL + Qwen2.5-VL") as demo:
        gr.Markdown("# ✍️ Universal Handwritten Notes & Document Studio", elem_id="main-title")
        gr.Markdown(
            "Reads **ANY File** (PDF, Images, DOCX, TXT) ➔ Converts to **Standard PDF** ➔ AI Reasoning (PaddleOCR-VL + Qwen2.5-VL)",
            elem_id="subtitle"
        )

        with gr.Row():
            # ================= Left Column: Universal Ingestion & Controls =================
            with gr.Column(scale=1):
                doc_file = gr.File(
                    label="📂 Upload Any File (PDF, Images, Word DOCX, TXT, TIFF, etc.)",
                    file_types=[
                        ".pdf",
                        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif",
                        ".docx",
                        ".txt", ".md", ".csv"
                    ]
                )

                with gr.Accordion("📷 Or Capture from Camera / Clipboard", open=False):
                    camera_image = gr.Image(sources=["webcam", "clipboard"], type="pil", label="Capture Direct Image")

                with gr.Row():
                    doc_status = gr.Markdown("No document loaded")
                    doc_scope = gr.Radio(
                        choices=["Current Selected Page", "All Pages (Entire Document)"],
                        value="Current Selected Page",
                        label="Processing Scope"
                    )

                with gr.Row():
                    page_slider = gr.Slider(
                        minimum=1, maximum=10, step=1, value=1, label="Page Selector", interactive=False
                    )
                    doc_dpi = gr.Slider(
                        minimum=100, maximum=300, step=25, value=150, label="Rendering DPI (Quality)"
                    )

                with gr.Row():
                    prev_page_btn = gr.Button("◀ Prev Page", size="sm")
                    next_page_btn = gr.Button("Next Page ▶", size="sm")

                page_preview = gr.Image(type="pil", label="Page Preview (Standard PDF View)", interactive=False)

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

                with gr.Accordion("🎯 Accuracy & Optimization", open=True):
                    auto_enhance = gr.Checkbox(
                        label="Auto-Enhance Contrast & Sharpness (Optimal for faint pencil/ink notes)",
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

                with gr.Accordion("⚙️ Hardware & Model Lifecycle", open=False):
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
                        info="For 8GB RAM, 'Qwen/Qwen2.5-VL-3B-Instruct' is recommended."
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=256, maximum=3072, value=1536, step=128, label="Max New Tokens"
                    )
                    with gr.Row():
                        btn_reload_models = gr.Button("🔄 Reload Models", size="sm", variant="secondary")
                        btn_unload_models = gr.Button("🧹 Free RAM (Unload)", size="sm")
                    model_status_log = gr.Markdown("Models load automatically upon analysis.")

                with gr.Row():
                    run_btn = gr.Button("🚀 Run Analysis (Background)", variant="primary", size="lg", scale=3)
                    stop_btn = gr.Button("⏹ Stop", variant="stop", size="lg", scale=1)

                live_status = gr.Markdown("**Status:** Idle", elem_classes=["status-box"])

            # ================= Right Column: Results & Dual Export =================
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("🌟 Final Verified Result & PDF Export"):
                        final_output = gr.Markdown(label="Final Transcription")
                        with gr.Row():
                            download_md_btn = gr.Button("💾 Download as Markdown (.md)", size="sm")
                            download_pdf_btn = gr.Button("📄 Download as Standard PDF (.pdf)", size="sm", variant="primary")
                        with gr.Row():
                            download_file_md = gr.File(label="Download Markdown", visible=False)
                            download_file_pdf = gr.File(label="Download PDF", visible=False)

                    with gr.TabItem("📊 Multi-Model Raw Extracts"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### 🟢 PaddleOCR-VL (Optical Characters)")
                                paddle_raw_output = gr.Textbox(label="PaddleOCR Output", lines=20)
                            with gr.Column():
                                gr.Markdown("#### 🔵 Qwen2.5-VL (Visual Reasoning)")
                                qwen_raw_output = gr.Textbox(label="Qwen2.5-VL Output", lines=20)

        # ================= UI Event Wiring =================
        def on_doc_uploaded(file, dpi, enhance):
            try:
                raw_path = get_file_path(file)
                if not raw_path or not os.path.exists(raw_path):
                    return gr.update(minimum=1, maximum=10, value=1, interactive=False), "⚠️ No file received", None

                ext = os.path.splitext(raw_path)[1].upper()
                pdf_path = convert_any_file_to_standard_pdf(raw_path)
                if not pdf_path or not os.path.exists(pdf_path):
                    return gr.update(minimum=1, maximum=10, value=1, interactive=False), f"⚠️ Could not convert {ext} file to PDF", None

                total = get_pdf_page_count(pdf_path)
                if total == 0:
                    return gr.update(minimum=1, maximum=10, value=1, interactive=False), "⚠️ Document has 0 readable pages", None

                preview = render_pdf_page(pdf_path, 0, dpi=int(dpi or 150), enhance=bool(enhance))
                slider_max = max(2, total)
                return (
                    gr.update(minimum=1, maximum=slider_max, value=1, interactive=(total > 1)),
                    f"✅ **Converted to Standard PDF:** {total} page(s) ({ext})",
                    preview
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                return gr.update(minimum=1, maximum=10, value=1, interactive=False), f"⚠️ Error loading document: {str(e)}", None

        doc_file.upload(
            fn=on_doc_uploaded,
            inputs=[doc_file, doc_dpi, auto_enhance],
            outputs=[page_slider, doc_status, page_preview]
        )

        doc_file.clear(
            fn=lambda: (gr.update(minimum=1, maximum=10, value=1, interactive=False), "No document loaded", None),
            outputs=[page_slider, doc_status, page_preview]
        )

        def on_page_changed(file, page_num, dpi, enhance):
            try:
                raw_path = get_file_path(file)
                if not raw_path:
                    return None
                pdf_path = convert_any_file_to_standard_pdf(raw_path)
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
            inputs=[doc_file, page_slider, doc_dpi, auto_enhance],
            outputs=[page_preview]
        )

        doc_dpi.change(
            fn=on_page_changed,
            inputs=[doc_file, page_slider, doc_dpi, auto_enhance],
            outputs=[page_preview]
        )

        def navigate_page(file, current_page, direction):
            try:
                raw_path = get_file_path(file)
                if not raw_path:
                    return 1
                pdf_path = convert_any_file_to_standard_pdf(raw_path)
                total = get_pdf_page_count(pdf_path)
                curr = int(current_page or 1)
                new_page = max(1, min(total, curr + direction))
                return new_page
            except Exception:
                return 1

        prev_page_btn.click(
            fn=lambda f, p: navigate_page(f, p, -1),
            inputs=[doc_file, page_slider],
            outputs=[page_slider]
        )

        next_page_btn.click(
            fn=lambda f, p: navigate_page(f, p, 1),
            inputs=[doc_file, page_slider],
            outputs=[page_slider]
        )

        prompt_templates.change(
            fn=lambda x, cur: x if x != "Custom" else cur,
            inputs=[prompt_templates, custom_prompt_input],
            outputs=[custom_prompt_input]
        )

        btn_reload_models.click(
            fn=reload_models,
            inputs=[paddle_path_input, qwen_path_input],
            outputs=[model_status_log]
        )

        btn_unload_models.click(
            fn=unload_all_models,
            inputs=[],
            outputs=[model_status_log]
        )

        run_event = run_btn.click(
            fn=execute_pipeline,
            inputs=[
                doc_file,
                camera_image,
                page_slider,
                doc_scope,
                doc_dpi,
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

        download_md_btn.click(
            fn=lambda text: gr.update(value=export_markdown_file(text), visible=True) if text else gr.update(visible=False),
            inputs=[final_output],
            outputs=[download_file_md]
        )

        download_pdf_btn.click(
            fn=lambda text: gr.update(value=export_standard_pdf(text), visible=True) if text else gr.update(visible=False),
            inputs=[final_output],
            outputs=[download_file_pdf]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CUSTOM_CSS)

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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re

# =====================================================================
# Font Registration for Indic (Marathi / Devanagari) + English PDF Export
# =====================================================================
INDIC_FONT_NAME = "Helvetica"
INDIC_FONT_BOLD = "Helvetica-Bold"

try:
    windows_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    nirmala_path = os.path.join(windows_fonts, "Nirmala.ttc")
    if os.path.exists(nirmala_path):
        pdfmetrics.registerFont(TTFont("Nirmala", nirmala_path, subfontIndex=0))
        pdfmetrics.registerFont(TTFont("NirmalaBold", nirmala_path, subfontIndex=1))
        INDIC_FONT_NAME = "Nirmala"
        INDIC_FONT_BOLD = "NirmalaBold"
        print("[Font Setup] Successfully registered Nirmala TTC for Marathi Devanagari + English typography.")
    else:
        mangal_path = os.path.join(windows_fonts, "mangal.ttf")
        if os.path.exists(mangal_path):
            pdfmetrics.registerFont(TTFont("Mangal", mangal_path))
            INDIC_FONT_NAME = "Mangal"
            INDIC_FONT_BOLD = "Mangal"
            print("[Font Setup] Successfully registered Mangal TTF for Marathi Devanagari typography.")
except Exception as fe:
    print(f"[Font Setup Warning] Could not register Indic font, falling back to Helvetica: {fe}")


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
    Accepts ANY file format (PDF, PNG, JPG, WEBP, TIFF, BMP, GIF, ICO, DOCX, TXT, MD, CSV, JSON, Code, etc.)
    and normalizes it into a standard, paginated PDF file path.
    """
    if not file_path or not os.path.exists(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()
    temp_dir = tempfile.gettempdir()
    out_pdf_path = os.path.join(temp_dir, f"normalized_{os.path.basename(file_path)}.pdf")

    # 1. Already a standard PDF
    if ext == ".pdf":
        return file_path

    # 2. All Image Formats (Single & Multi-page TIFF, PNG, JPG, JPEG, WEBP, BMP, GIF, ICO, PPM, etc.)
    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif", ".ico", ".ppm", ".pgm", ".pbm"]:
        try:
            im = Image.open(file_path)
            frames = []
            try:
                while True:
                    frames.append(im.copy().convert("RGB"))
                    im.seek(len(frames))
            except EOFError:
                pass

            if len(frames) > 1:
                frames[0].save(out_pdf_path, "PDF", resolution=150.0, save_all=True, append_images=frames[1:])
            elif frames:
                frames[0].save(out_pdf_path, "PDF", resolution=150.0)
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

    # 3. Word Documents (.docx, .doc)
    if ext in [".docx", ".doc"] and HAS_DOCX:
        try:
            doc = docx.Document(file_path)
            styles = getSampleStyleSheet()
            h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName=INDIC_FONT_BOLD, fontSize=16, spaceAfter=8, textColor=colors.HexColor('#1e40af'))
            body = ParagraphStyle('Body', parent=styles['Normal'], fontName=INDIC_FONT_NAME, fontSize=10, leading=15, spaceAfter=4)

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

    # 4. Plain Text, Markdown, CSV, TSV, Code & Data files (.txt, .md, .csv, .tsv, .json, .xml, .yaml, .yml, .py, .java, .cpp, .c, .html, .log, etc.)
    text_extensions = [
        ".txt", ".md", ".csv", ".tsv", ".log", ".json", ".xml", ".yaml", ".yml",
        ".py", ".js", ".ts", ".html", ".htm", ".css", ".java", ".c", ".cpp",
        ".h", ".sql", ".ini", ".cfg", ".sh", ".bat", ".ps1", ".rtf"
    ]
    if ext in text_extensions:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return export_standard_pdf(content, out_pdf_path)
        except Exception as e:
            print(f"Text/code to PDF conversion error: {e}")

    # 5. Generic PyMuPDF fallback for other document formats (XPS, EPUB, MOBI, SVG, etc.)
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
        # Universal & crash-proof conversion for Grayscale, CMYK, and RGB pages
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
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
# High-Accuracy Prompts (Marathi + English Bilingual & General)
# =====================================================================
MARATHI_ENGLISH_PROMPT_TEMPLATE = """You are an elite bilingual handwriting transcription specialist, expert in Marathi (मराठी - देवनागरी लिपी) and English handwritten documents, academic lecture notes, and official papers.

PRIMARY INPUTS:
1. Visual scan of the handwritten note/document.
2. Raw optical characters & formula extract from PaddleOCR-VL:
--- RAW OCR EXTRACT ---
{paddle_text}
--- END RAW OCR EXTRACT ---

USER SPECIFIC INSTRUCTIONS:
{user_prompt}

STRICT BILINGUAL & STANDARD TYPEWRITING CONVERSION RULES:
1. MARATHI (मराठी) LANGUAGE ACCURACY:
   - Faithfully transcribe all Marathi handwritten text in standard Devanagari script (मराठी देवनागरी लिपी).
   - Accurately reproduce all conjuncts (जोडअक्षरे: जसे की प्र, त्र, ज्ञ, क्ष, द्ध, ष्ट, ण्य, ळ्ळ, रफार).
   - Preserve all matras, kanas, velantis, and ukaars (काना, र्‍हस्व/दीर्घ वेलांटी, उकार, मात्रा, अनुस्वार).
   - Follow standard Marathi orthography and grammar rules (महाराष्ट्र राज्य साहित्य महामंडळाचे अधिकृत शुद्धलेखन नियम).
   - Recognize Marathi numerals (०, १, २, ३, ४, ५, ६, ७, ८, ९) or Arabic numerals (0-9) exactly as written.

2. ENGLISH LANGUAGE ACCURACY:
   - Accurately transcribe all English cursive, print, and mixed handwriting.
   - Resolve ambiguous letter shapes based on surrounding context (e.g., distinguishing '0' vs 'O', '1' vs 'l' vs 'I', '5' vs 'S', 'q' vs 'g').
   - Keep technical terms, acronyms, and scientific names in standard English.

3. CONVERSION TO "STANDARD TYPEWRITING FORMAT" (प्रमाणित टंकलेखन स्वरूप):
   - Convert messy, hasty, or slanted handwriting into clean, formal, publication-ready typewritten text.
   - Use clear hierarchical Markdown headings:
     # मुख्य शीर्षक (Main Title)
     ## उपशीर्षक (Section / Topic)
     ### तपशील (Subtopic / Subsection)
   - Format definitions, key points, and summaries with clear bullet points (- ) or numbered lists (1. , 2. ).
   - If handwritten text includes tabular data, format it into an aligned Markdown table:
     | अनुक्रमांक / Sr. No. | तपशील / Description | मूल्य / Value |
   - Format callout boxes, notes, or tips with blockquotes (> ...).

4. MATHEMATICAL & SCIENTIFIC FORMULAS:
   - Convert all handwritten mathematical, statistical, or chemical expressions into standard LaTeX.
   - Use $...$ for inline math (e.g., $f(x) = \\frac{{1}}{{\\sigma\\sqrt{{2\\pi}}}}e^{{-\\frac{{(x-\\mu)^2}}{{2\\sigma^2}}}}$).
   - Use $$...$$ on a separate line for standalone display formulas.

5. ZERO HALLUCINATION POLICY:
   - Never fabricate or guess unwritten content.
   - If a word is torn, faded, or completely unreadable, mark it clearly as [अस्पष्ट / Unclear] rather than guessing.

Now transcribe the handwritten note into clean, standard typewriting format below:"""


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
            is_marathi = any(kw in (custom_prompt or "").lower() for kw in ["marathi", "मराठी", "devanagari", "देवनागरी", "bilingual", "टंकलेखन"]) or not custom_prompt.strip()
            template = MARATHI_ENGLISH_PROMPT_TEMPLATE if is_marathi else HIGH_ACCURACY_PROMPT_TEMPLATE
            full_prompt = template.format(
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
        from reportlab.platypus import Table, TableStyle

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'DocTitle', parent=styles['Heading1'],
            fontName=INDIC_FONT_BOLD, fontSize=18, leading=23,
            spaceAfter=10, textColor=colors.HexColor('#0f172a')
        )
        h1_style = ParagraphStyle(
            'DocH1', parent=styles['Heading1'],
            fontName=INDIC_FONT_BOLD, fontSize=14, leading=19,
            spaceAfter=8, textColor=colors.HexColor('#1e40af')
        )
        h2_style = ParagraphStyle(
            'DocH2', parent=styles['Heading2'],
            fontName=INDIC_FONT_BOLD, fontSize=12, leading=17,
            spaceAfter=6, textColor=colors.HexColor('#2563eb')
        )
        body_style = ParagraphStyle(
            'DocBody', parent=styles['Normal'],
            fontName=INDIC_FONT_NAME, fontSize=10, leading=15,
            spaceAfter=4, textColor=colors.HexColor('#334155')
        )
        bullet_style = ParagraphStyle(
            'DocBullet', parent=styles['Normal'],
            fontName=INDIC_FONT_NAME, fontSize=10, leading=15,
            spaceAfter=3, leftIndent=15, textColor=colors.HexColor('#334155')
        )
        quote_style = ParagraphStyle(
            'DocQuote', parent=styles['Normal'],
            fontName=INDIC_FONT_NAME, fontSize=9.5, leading=14,
            spaceAfter=5, leftIndent=20, textColor=colors.HexColor('#475569')
        )
        table_header_style = ParagraphStyle(
            'TH', parent=styles['Normal'],
            fontName=INDIC_FONT_BOLD, fontSize=9.5, leading=13,
            textColor=colors.white
        )
        table_cell_style = ParagraphStyle(
            'TD', parent=styles['Normal'],
            fontName=INDIC_FONT_NAME, fontSize=9, leading=13,
            textColor=colors.HexColor('#1e293b')
        )

        def clean_md_inline(txt):
            txt = txt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            txt = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', txt)
            txt = re.sub(r'\*(.+?)\*', r'<i>\1</i>', txt)
            return txt

        story = []
        lines = markdown_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                story.append(Spacer(1, 4))
                i += 1
                continue

            # Table parsing
            if line.startswith('|') and line.endswith('|'):
                table_rows = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    row_line = lines[i].strip()
                    if re.match(r'^\|[\s\-:|]+\|$', row_line):
                        i += 1
                        continue
                    cells = [c.strip() for c in row_line.split('|')[1:-1]]
                    if not table_rows:
                        table_rows.append([Paragraph(clean_md_inline(c), table_header_style) for c in cells])
                    else:
                        table_rows.append([Paragraph(clean_md_inline(c), table_cell_style) for c in cells])
                    i += 1
                if table_rows:
                    num_cols = max(len(r) for r in table_rows)
                    col_w = 480.0 / max(1, num_cols)
                    col_widths = [col_w] * num_cols
                    t = Table(table_rows, colWidths=col_widths)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e40af')),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                        ('TOPPADDING', (0,0), (-1,-1), 5),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8fafc'), colors.white])
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 6))
                continue

            if line.startswith('# '):
                story.append(Paragraph(clean_md_inline(line[2:]), title_style))
            elif line.startswith('## '):
                story.append(Paragraph(clean_md_inline(line[3:]), h1_style))
            elif line.startswith('### '):
                story.append(Paragraph(clean_md_inline(line[4:]), h2_style))
            elif line.startswith('---'):
                story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#cbd5e1'), spaceAfter=6))
            elif line.startswith('> '):
                story.append(Paragraph(clean_md_inline(line[2:]), quote_style))
            elif line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
                bullet_text = clean_md_inline(line[2:])
                story.append(Paragraph(f"&bull;&nbsp;&nbsp;{bullet_text}", bullet_style))
            elif re.match(r'^\d+[\.\)]\s+', line):
                story.append(Paragraph(clean_md_inline(line), bullet_style))
            elif line.startswith('```'):
                i += 1
                continue
            else:
                story.append(Paragraph(clean_md_inline(line), body_style))
            i += 1

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
    with gr.Blocks(title="Handwritten Notes Studio: Marathi & English Bilingual (PaddleOCR-VL + Qwen2.5-VL)") as demo:
        gr.Markdown("# ✍️ Marathi & English Handwritten Notes Studio", elem_id="main-title")
        gr.Markdown(
            "Reads **EVERY File Format** ➔ Converts to **Standard PDF** ➔ **Marathi (मराठी) & English** AI Typewriting Format",
            elem_id="subtitle"
        )

        with gr.Row():
            # ================= Left Column: Universal Ingestion & Controls =================
            with gr.Column(scale=1):
                doc_file = gr.File(
                    label="📂 Upload Any File (PDF, Images, DOCX, TXT, Multi-page TIFF, etc.)",
                    file_types=[
                        ".pdf",
                        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif", ".ico",
                        ".docx", ".doc",
                        ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".log", ".py"
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

                with gr.Accordion("🎯 Accuracy & Bilingual Optimization", open=True):
                    auto_enhance = gr.Checkbox(
                        label="Auto-Enhance Contrast & Sharpness (Optimal for faint pencil/ink notes)",
                        value=True
                    )
                    prompt_templates = gr.Dropdown(
                        label="Task Preset",
                        choices=[
                            "Transcribe the handwritten notes accurately in Marathi (मराठी देवनागरी लिपी) and English into standard typewritten format with LaTeX math and clean markdown headings.",
                            "Transcribe the handwritten notes into structured Markdown and LaTeX formulas.",
                            "Verbatim exact transcription preserving original handwritten layout (शब्दशः मुद्रण).",
                            "Lecture & Study Notes Explainer (टिपण्या व सविस्तर स्पष्टीकरण).",
                            "Extract all math equations, definitions, and theorems from the notes.",
                            "Custom"
                        ],
                        value="Transcribe the handwritten notes accurately in Marathi (मराठी देवनागरी लिपी) and English into standard typewritten format with LaTeX math and clean markdown headings."
                    )
                    custom_prompt_input = gr.Textbox(
                        label="Custom Instruction",
                        value="Transcribe the handwritten notes accurately in Marathi (मराठी देवनागरी लिपी) and English into standard typewritten format with LaTeX math and clean markdown headings.",
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

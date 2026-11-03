"""
OpenAI-Compatible REST API Server for Handwritten Notes Studio
Bilingual Marathi (मराठी) + English (इंग्रजी) & Universal Document Engine

Features:
 - Full OpenAI /v1/chat/completions API specification (Streaming SSE + JSON)
 - Multimodal Vision support (Base64 data URLs, image URLs, local paths)
 - Universal Document Ingestion: Any file format -> Standard PDF
 - Marathi & English handwriting to Standard Typewriting Format
 - Direct PDF and Markdown export endpoints
 - Mobile Scanner web interface at http://<ip>:8000/
 - Swagger API documentation at http://<ip>:8000/docs
"""

import os
import sys

# Disable TensorFlow/OneDNN warnings & force pure PyTorch
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gc
import io
import time
import uuid
import json
import base64
import tempfile
import threading
from threading import Thread
from typing import List, Optional, Union, Dict, Any

import torch
from PIL import Image, ImageOps, ImageEnhance
import pymupdf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import pipeline functions from app_merged
from app_merged import (
    loaded_models,
    DEVICE,
    DTYPE,
    DEFAULT_PADDLE_PATH,
    DEFAULT_QWEN_PATH,
    load_paddle_model,
    load_qwen_model,
    unload_paddle_model,
    unload_all_models,
    run_paddle_ocr,
    stream_qwen_vl,
    convert_any_file_to_standard_pdf,
    render_pdf_page,
    get_pdf_page_count,
    enhance_handwritten_image,
    export_standard_pdf,
    export_markdown_file,
    MARATHI_ENGLISH_PROMPT_TEMPLATE,
    HIGH_ACCURACY_PROMPT_TEMPLATE,
    INDIC_FONT_NAME
)

app = FastAPI(
    title="Handwritten Notes Studio - OpenAI API Server",
    description="OpenAI-compatible Vision & Document API for Marathi and English Handwritten Notes.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Pydantic Schemas (OpenAI-compatible)
# =====================================================================
class ChatMessageContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], List[ChatMessageContentPart]]


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-vl"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 1536
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "handwritten-notes-studio"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# =====================================================================
# Helper: Extract Text & Image from Multimodal OpenAI Messages
# =====================================================================
def extract_text_and_image(messages: List[ChatMessage]):
    system_prompt = ""
    user_prompt = ""
    pil_image = None

    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_prompt += msg.content + "\n"
        elif msg.role in ["user", "assistant"]:
            if isinstance(msg.content, str):
                user_prompt += msg.content + "\n"
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict):
                        p_type = part.get("type")
                        if p_type == "text":
                            user_prompt += part.get("text", "") + "\n"
                        elif p_type == "image_url":
                            img_obj = part.get("image_url", {})
                            url = img_obj.get("url") if isinstance(img_obj, dict) else img_obj
                            if url and pil_image is None:
                                pil_image = decode_image_from_url(url)
                    elif hasattr(part, "type"):
                        if part.type == "text" and part.text:
                            user_prompt += part.text + "\n"
                        elif part.type == "image_url" and part.image_url:
                            url = part.image_url.get("url") if isinstance(part.image_url, dict) else str(part.image_url)
                            if url and pil_image is None:
                                pil_image = decode_image_from_url(url)

    combined_prompt = f"{system_prompt.strip()}\n{user_prompt.strip()}".strip()
    return combined_prompt, pil_image


def decode_image_from_url(url: str) -> Optional[Image.Image]:
    try:
        if url.startswith("data:image"):
            # Base64 data URL
            header, base64_data = url.split(",", 1)
            image_data = base64.b64decode(base64_data)
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        elif os.path.exists(url):
            # Local file path
            return Image.open(url).convert("RGB")
        else:
            # Remote URL
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


# =====================================================================
# OpenAI Endpoints
# =====================================================================
@app.get("/v1/models", response_model=ModelList)
def list_models():
    models = [
        ModelCard(id="qwen2.5-vl"),
        ModelCard(id="paddleocr-vl"),
        ModelCard(id="handwritten-bilingual"),
        ModelCard(id="marathi-typewriter")
    ]
    return ModelList(data=models)


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    prompt, image = extract_text_and_image(request.messages)

    if not prompt and image is None:
        raise HTTPException(status_code=400, detail="Either text prompt or image must be provided.")

    is_marathi = any(kw in prompt.lower() for kw in ["marathi", "मराठी", "देवनागरी", "devanagari", "bilingual", "टंकलेखन"]) or not prompt.strip()

    # If image provided, optionally run PaddleOCR-VL optical pre-pass for high accuracy
    paddle_text = ""
    if image is not None:
        image = enhance_handwritten_image(image)
        if request.model in ["paddleocr-vl"]:
            if loaded_models["paddle"]["model"] is None:
                load_paddle_model(DEFAULT_PADDLE_PATH)
            paddle_text = run_paddle_ocr(image, prompt="OCR:", max_tokens=request.max_tokens or 1024)
            return JSONResponse(content={
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": paddle_text},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": len(paddle_text.split()), "total_tokens": 100 + len(paddle_text.split())}
            })

        # Dual-pipeline pre-extraction
        if loaded_models["paddle"]["model"] is not None:
            paddle_text = run_paddle_ocr(image, prompt="OCR:", max_tokens=1024)

    # Format bilingual or general prompt
    if is_marathi:
        template = MARATHI_ENGLISH_PROMPT_TEMPLATE
    else:
        template = HIGH_ACCURACY_PROMPT_TEMPLATE

    if "{paddle_text}" in template:
        final_prompt = template.format(paddle_text=paddle_text, user_prompt=prompt)
    else:
        final_prompt = f"{prompt}\n{paddle_text}".strip()

    # Ensure Qwen model is ready
    if loaded_models["qwen"]["model"] is None:
        load_qwen_model(DEFAULT_QWEN_PATH)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Streaming mode (SSE)
    if request.stream:
        def stream_generator():
            created_ts = int(time.time())
            for chunk in stream_qwen_vl(image, prompt=final_prompt, max_tokens=request.max_tokens or 1536):
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            # Final chunk
            stop_data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(stop_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Non-streaming mode
    accumulated_response = ""
    for chunk in stream_qwen_vl(image, prompt=final_prompt, max_tokens=request.max_tokens or 1536):
        accumulated_response = chunk

    return JSONResponse(content={
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": accumulated_response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(final_prompt.split()),
            "completion_tokens": len(accumulated_response.split()),
            "total_tokens": len(final_prompt.split()) + len(accumulated_response.split())
        }
    })


# =====================================================================
# Document Processing & Mobile Helper Endpoints
# =====================================================================
@app.post("/v1/documents/convert")
async def convert_document(file: UploadFile = File(...)):
    """Upload any file format and convert to standard PDF."""
    temp_dir = tempfile.gettempdir()
    input_path = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex[:6]}_{file.filename}")
    
    with open(input_path, "wb") as f:
        f.write(await file.read())

    pdf_path = convert_any_file_to_standard_pdf(input_path)
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Failed to convert file to standard PDF.")

    pages = get_pdf_page_count(pdf_path)
    pdf_filename = os.path.basename(pdf_path)

    return {
        "status": "success",
        "original_file": file.filename,
        "pdf_pages": pages,
        "pdf_filename": pdf_filename,
        "download_url": f"/v1/download/{pdf_filename}"
    }


@app.post("/v1/documents/export-pdf")
async def export_pdf_endpoint(request: Request):
    """Generates standard publication PDF with Marathi Devanagari font."""
    body = await request.json()
    markdown_text = body.get("markdown", "")
    if not markdown_text:
        raise HTTPException(status_code=400, detail="Missing markdown content.")

    temp_dir = tempfile.gettempdir()
    out_filename = f"transcribed_{uuid.uuid4().hex[:8]}.pdf"
    out_path = os.path.join(temp_dir, out_filename)

    pdf_path = export_standard_pdf(markdown_text, out_path)
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Error generating PDF.")

    return {
        "status": "success",
        "pdf_filename": out_filename,
        "download_url": f"/v1/download/{out_filename}"
    }


@app.get("/v1/download/{filename}")
def download_file(filename: str):
    """Safely serves generated PDF/Markdown files."""
    temp_dir = tempfile.gettempdir()
    safe_path = os.path.abspath(os.path.join(temp_dir, filename))
    if not safe_path.startswith(temp_dir) or not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail="File not found.")
    
    media_type = "application/pdf" if filename.endswith(".pdf") else "text/markdown"
    return FileResponse(safe_path, media_type=media_type, filename=filename)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "dtype": str(DTYPE),
        "paddle_loaded": loaded_models["paddle"]["model"] is not None,
        "qwen_loaded": loaded_models["qwen"]["model"] is not None,
        "indic_font": INDIC_FONT_NAME
    }


# =====================================================================
# Mobile Scanner Web Client (Served at /)
# =====================================================================
MOBILE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>✍️ Handwritten Notes Studio Mobile</title>
  <style>
    :root {
      --primary: #2563eb;
      --bg: #0f172a;
      --card: #1e293b;
      --text: #f8fafc;
      --accent: #38bdf8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 12px; }
    .header { text-align: center; margin-bottom: 16px; padding-top: 8px; }
    .header h1 { font-size: 1.3rem; color: #fff; }
    .header p { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .card { background: var(--card); border-radius: 12px; padding: 14px; margin-bottom: 14px; border: 1px solid #334155; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
    .btn { display: flex; align-items: center; justify-content: center; gap: 6px; padding: 12px; border-radius: 8px; font-weight: 600; font-size: 0.9rem; cursor: pointer; border: none; text-align: center; }
    .btn-camera { background: #16a34a; color: white; }
    .btn-upload { background: var(--primary); color: white; }
    .btn-submit { background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; width: 100%; padding: 14px; margin-top: 10px; font-size: 1rem; }
    .btn-pdf { background: #dc2626; color: white; width: 100%; padding: 12px; margin-top: 10px; text-decoration: none; display: none; text-align: center; border-radius: 8px; font-weight: 600; }
    input[type="file"] { display: none; }
    .preview-box { width: 100%; max-height: 220px; border-radius: 8px; overflow: hidden; display: none; margin-bottom: 10px; text-align: center; background: #000; }
    .preview-box img { max-width: 100%; max-height: 220px; object-fit: contain; }
    select, textarea { width: 100%; background: #0f172a; color: #fff; border: 1px solid #475569; border-radius: 8px; padding: 10px; margin-top: 6px; font-size: 0.9rem; }
    .output-box { min-height: 140px; max-height: 350px; overflow-y: auto; background: #020617; border-radius: 8px; padding: 12px; font-size: 0.88rem; line-height: 1.5; white-space: pre-wrap; border: 1px solid #334155; }
    .badge { display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; background: #334155; color: #38bdf8; margin-bottom: 6px; }
  </style>
</head>
<body>
  <div class="header">
    <h1>✍️ Handwritten Notes Studio</h1>
    <p>मराठी (Devanagari) & English Mobile Scanner</p>
  </div>

  <div class="card">
    <div class="actions">
      <label class="btn btn-camera">
        📸 Camera
        <input type="file" id="camera-input" accept="image/*" capture="environment">
      </label>
      <label class="btn btn-upload">
        📂 Upload
        <input type="file" id="file-input" accept="image/*,.pdf,.docx,.txt">
      </label>
    </div>

    <div class="preview-box" id="preview-box">
      <img id="preview-img" src="">
    </div>

    <label style="font-size: 0.85rem; color: #cbd5e1;">Language & Format Preset:</label>
    <select id="preset-select">
      <option value="marathi_bilingual">🇮🇳 मराठी आणि इंग्रजी (Marathi & English Bilingual)</option>
      <option value="standard_latex">📐 Math & Formulas (LaTeX)</option>
      <option value="verbatim">📄 Verbatim Exact Transcription</option>
    </select>

    <button class="btn btn-submit" id="process-btn">🚀 Convert to Typewriting Format</button>
  </div>

  <div class="card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
      <span class="badge" id="status-badge">Ready</span>
      <span style="font-size: 0.75rem; color: #94a3b8;" id="timer"></span>
    </div>
    <div class="output-box" id="output-box">Transcription will stream here live...</div>
    <a href="#" class="btn-pdf" id="download-pdf-btn">📄 Download Standard PDF</a>
  </div>

  <script>
    let selectedFile = null;
    let base64Image = null;

    function handleFile(file) {
      if (!file) return;
      selectedFile = file;
      document.getElementById('status-badge').innerText = 'Selected: ' + file.name;
      
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          base64Image = e.target.result;
          document.getElementById('preview-img').src = base64Image;
          document.getElementById('preview-box').style.display = 'block';
        };
        reader.readAsDataURL(file);
      } else {
        document.getElementById('preview-box').style.display = 'none';
      }
    }

    document.getElementById('camera-input').addEventListener('change', (e) => handleFile(e.target.files[0]));
    document.getElementById('file-input').addEventListener('change', (e) => handleFile(e.target.files[0]));

    document.getElementById('process-btn').addEventListener('click', async () => {
      if (!base64Image && !selectedFile) {
        alert('Please snap a photo or select a file first.');
        return;
      }

      const outputBox = document.getElementById('output-box');
      const badge = document.getElementById('status-badge');
      const pdfBtn = document.getElementById('download-pdf-btn');
      outputBox.innerText = 'Analyzing handwriting in Marathi & English...';
      badge.innerText = 'Processing...';
      pdfBtn.style.display = 'none';

      try {
        const payload = {
          model: "handwritten-bilingual",
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "Transcribe this handwritten document in Marathi and English into clean standard typewriting format." },
                { type: "image_url", image_url: { url: base64Image } }
              ]
            }
          ],
          stream: true
        };

        const res = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        outputBox.innerText = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split('\\n');
          for (const line of lines) {
            if (line.startsWith('data: ') && !line.includes('[DONE]')) {
              try {
                const parsed = JSON.parse(line.substring(6));
                const content = parsed.choices[0]?.delta?.content || '';
                outputBox.innerText += content;
                outputBox.scrollTop = outputBox.scrollHeight;
              } catch (e) {}
            }
          }
        }

        badge.innerText = 'Completed';

        // Request standard PDF generation
        const pdfRes = await fetch('/v1/documents/export-pdf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ markdown: outputBox.innerText })
        });
        const pdfData = await pdfRes.json();
        if (pdfData.download_url) {
          pdfBtn.href = pdfData.download_url;
          pdfBtn.style.display = 'block';
        }
      } catch (err) {
        outputBox.innerText = 'Error: ' + err.message;
        badge.innerText = 'Error';
      }
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_mobile_home():
    return HTMLResponse(content=MOBILE_HTML)


if __name__ == "__main__":
    import uvicorn
    print("=================================================================")
    print("🚀 Handwritten Notes Studio - OpenAI REST API Server")
    print("🌐 Endpoints:")
    print("  - OpenAI Completions:  http://127.0.0.1:8000/v1/chat/completions")
    print("  - Model Catalog:       http://127.0.0.1:8000/v1/models")
    print("  - Mobile Scanner Web:  http://127.0.0.1:8000/")
    print("  - Interactive Swagger: http://127.0.0.1:8000/docs")
    print("=================================================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)

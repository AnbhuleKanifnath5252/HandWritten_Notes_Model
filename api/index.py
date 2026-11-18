from http.server import BaseHTTPRequestHandler
import json
import urllib.parse

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
        
        if path.endswith("/health"):
            data = {
                "status": "healthy",
                "service": "Handwritten Notes Studio",
                "version": "2.5.0",
                "database_required": False
            }
        elif path.endswith("/models"):
            data = {
                "models": [
                    {"id": "PaddleOCR-VL", "type": "OCR & Layout"},
                    {"id": "Qwen2.5-VL-7B-Instruct", "type": "Visual Reasoning & Transcription"}
                ],
                "database_required": False
            }
        elif "mathvision" in path:
            data = {
                "service": "MathVision Evaluation Service",
                "status": "online",
                "database_required": False,
                "capabilities": ["mathematical_reasoning", "geometry_parsing", "equation_transcription"]
            }
        elif "mmmu" in path:
            data = {
                "service": "MMMU Multimodal Benchmark",
                "status": "online",
                "database_required": False,
                "disciplines": ["art", "business", "science", "humanities", "tech"]
            }
        elif "odinw-13" in path or "odinw_13" in path:
            data = {
                "service": "ODinW-13 Object Detection",
                "status": "online",
                "database_required": False,
                "tasks": ["bounding_boxes", "zero_shot_grounding"]
            }
        elif "realworldqa" in path:
            data = {
                "service": "RealWorldQA Spatial Reasoning",
                "status": "online",
                "database_required": False,
                "tasks": ["spatial_understanding", "physical_world_qa"]
            }
        elif "videomme" in path:
            data = {
                "service": "VideoMME Video Comprehension",
                "status": "online",
                "database_required": False,
                "tasks": ["temporal_reasoning", "video_qa"]
            }
        elif "qwen-vl-utils" in path or "qwen_vl_utils" in path:
            data = {
                "service": "qwen-vl-utils Vision Preprocessor",
                "status": "online",
                "database_required": False,
                "functions": ["process_vision_info", "smart_resize"]
            }
        else:
            data = {
                "name": "Handwritten Notes Studio — Stateless Serverless API",
                "status": "online",
                "database_required": False,
                "font": "Times New Roman",
                "languages": ["English", "Marathi (मराठी)"],
                "services": [
                    "/api/health",
                    "/api/models",
                    "/services/mathvision",
                    "/services/mmmu",
                    "/services/odinw-13",
                    "/services/realworldqa",
                    "/services/videomme",
                    "/services/qwen-vl-utils"
                ]
            }
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else "{}"
        try:
            req = json.loads(body)
        except Exception:
            req = {}
            
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
        
        resp = {
            "status": "success",
            "database_required": False,
            "received_input": req,
            "message": "Processed successfully by Handwritten Notes Studio API (Times New Roman formatting enabled)"
        }
        self.wfile.write(json.dumps(resp, indent=2).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

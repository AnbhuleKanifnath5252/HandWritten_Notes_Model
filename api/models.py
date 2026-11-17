import json
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        data = {
            "object": "list",
            "data": [
                {
                    "id": "qwen2.5-vl",
                    "object": "model",
                    "owned_by": "handwritten-notes-studio",
                    "description": "Visual Reasoning & Transcription Engine"
                },
                {
                    "id": "paddleocr-vl",
                    "object": "model",
                    "owned_by": "handwritten-notes-studio",
                    "description": "Optical Character & Formula Detection"
                },
                {
                    "id": "handwritten-bilingual",
                    "object": "model",
                    "owned_by": "handwritten-notes-studio",
                    "description": "Marathi (मराठी) + English Dual-Model Engine"
                }
            ]
        }
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

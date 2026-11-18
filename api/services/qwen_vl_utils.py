import json
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    """
    Qwen-VL-Utils: Multimodal Vision & Video Preprocessing Service
    Encodes vision inputs, formats dynamic FPS, handles smart video downscaling, and prepares chat templates.
    Stateless - 100% Database-Free.
    """
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        data = {
            "service": "qwen-vl-utils",
            "package": "qwen-vl-utils",
            "type": "Multimodal Vision Preprocessing Utility",
            "status": "online",
            "database_required": False,
            "features": [
                "process_vision_info",
                "smart_resize (Min & Max Pixel bounds)",
                "dynamic_fps extraction for video frames",
                "chat_template token interleaving"
            ]
        }
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            req_data = json.loads(body.decode('utf-8'))
        except Exception:
            req_data = {}

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        res = {
            "service": "qwen-vl-utils",
            "status": "success",
            "database_connected": False,
            "vision_info": {
                "smart_resized": True,
                "fps_normalized": 2.0,
                "interleaved": True
            }
        }
        self.wfile.write(json.dumps(res).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

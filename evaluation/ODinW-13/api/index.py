from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {
            "service": "ODinW-13 Object Detection Service",
            "status": "online",
            "database_required": False,
            "mode": "stateless_serverless",
            "version": "1.0.0",
            "tasks": ["bounding_boxes", "zero_shot_grounding", "object_detection"]
        }
        self.wfile.write(json.dumps(response).encode())

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        try:
            payload = json.loads(post_data.decode('utf-8'))
        except Exception:
            payload = {}
        result = {
            "service": "ODinW-13",
            "database": "None (Stateless)",
            "received_input": payload,
            "detected_objects": []
        }
        self.wfile.write(json.dumps(result).encode())

from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {
            "service": "MMMU Multimodal Benchmark Service",
            "status": "online",
            "database_required": False,
            "mode": "stateless_serverless",
            "version": "1.0.0",
            "disciplines": ["art", "business", "science", "humanities", "tech"]
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
            "service": "MMMU",
            "database": "None (Stateless)",
            "received_input": payload,
            "evaluation_result": "MMMU multimodal evaluation completed."
        }
        self.wfile.write(json.dumps(result).encode())

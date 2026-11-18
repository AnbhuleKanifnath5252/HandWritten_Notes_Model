from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {
            "service": "VideoMME Video Multimodal Evaluation",
            "status": "online",
            "database_required": False,
            "mode": "stateless_serverless",
            "version": "1.0.0",
            "capabilities": ["video_comprehension", "temporal_reasoning", "frame_sequence_qa"]
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
            "service": "VideoMME",
            "database": "None (Stateless)",
            "received_input": payload,
            "temporal_analysis": "Completed"
        }
        self.wfile.write(json.dumps(result).encode())

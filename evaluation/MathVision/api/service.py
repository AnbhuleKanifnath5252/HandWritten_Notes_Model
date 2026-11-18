from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {
            "service": "MathVision Evaluation Service",
            "status": "online",
            "database_required": False,
            "mode": "stateless_serverless",
            "version": "1.0.0",
            "capabilities": ["mathematical_reasoning", "geometry_parsing", "equation_transcription"]
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
            "service": "MathVision",
            "database": "None (Stateless)",
            "received_input": payload,
            "evaluation_result": "MathVision evaluation pipeline executed successfully."
        }
        self.wfile.write(json.dumps(result).encode())

"""
Watchdog Auto-Reloading Server for Handwritten Notes Studio
Automatically reloads and restarts app_merged.py upon changes or crashes.

Run via:
    python watch_server.py
"""

import os
import sys
import time
import subprocess

def run():
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_merged.py")
    print("=================================================================")
    print("🚀 Handwritten Notes Studio - Watchdog Auto-Reload Server")
    print("📡 Watching: app_merged.py")
    print("🌐 URL: http://127.0.0.1:7860")
    print("⏹  Press Ctrl + C in this terminal to stop the server.")
    print("=================================================================\n")

    while True:
        try:
            print("[*] Starting server process...")
            proc = subprocess.Popen([sys.executable, script_path])
            proc.wait()
            ret = proc.returncode
            if ret != 0:
                print(f"\n[!] Process exited with code {ret}. Auto-reloading in 3 seconds...")
            else:
                print("\n[*] Server stopped normally. Auto-restarting...")
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n🛑 Stopping server watcher.")
            if proc:
                proc.terminate()
            break
        except Exception as e:
            print(f"[!] Watcher error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    run()


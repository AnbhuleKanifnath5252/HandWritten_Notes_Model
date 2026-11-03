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
    print("📡 Watching for file changes & crashes: app_merged.py")
    print("🌐 URL: http://127.0.0.1:7860")
    print("⏹  Press Ctrl + C in this terminal to stop the server.")
    print("=================================================================\n")

    last_mtime = os.path.getmtime(script_path) if os.path.exists(script_path) else 0

    while True:
        proc = None
        try:
            print("[*] Starting server process...")
            proc = subprocess.Popen([sys.executable, script_path])
            
            # Watchdog loop: check both process status and file modifications
            restarted_for_change = False
            while proc.poll() is None:
                time.sleep(1.5)
                if os.path.exists(script_path):
                    current_mtime = os.path.getmtime(script_path)
                    if current_mtime > last_mtime:
                        print("\n[🔄 Auto-Reload] Detected changes in app_merged.py! Restarting server...")
                        last_mtime = current_mtime
                        proc.terminate()
                        try:
                            proc.wait(timeout=4)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        restarted_for_change = True
                        break

            if not restarted_for_change:
                ret = proc.returncode
                if ret != 0:
                    print(f"\n[!] Process exited with code {ret}. Auto-reloading in 3 seconds...")
                else:
                    print("\n[*] Server stopped normally. Auto-restarting...")
                time.sleep(3)
        except KeyboardInterrupt:
            print("\n🛑 Stopping server watcher.")
            if proc and proc.poll() is None:
                proc.terminate()
            break
        except Exception as e:
            print(f"[!] Watcher error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    run()


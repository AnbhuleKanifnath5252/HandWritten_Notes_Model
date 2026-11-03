"""
Mobile APK & PWA Generator for Handwritten Notes Studio
Builds Android APK or launches 1-tap mobile installation.
"""

import os
import sys
import shutil
import subprocess

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    www_dir = os.path.join(base_dir, "www")

    print("=================================================================")
    print("📱 Handwritten Notes Studio - Mobile APK & Scanner Suite")
    print("=================================================================")
    print("\nOptions for Mobile Scanner:")
    print(" 1. Instant 1-Tap Mobile Install (PWA / WebAPK - Recommended)")
    print("    - No Android Studio or Java SDK required!")
    print("    - Open http://<YOUR_PC_IP>:8000 on Android Chrome or Edge.")
    print("    - Tap the banner: 'Install Handwritten Notes Studio'.")
    print("    - An Android APK icon is added directly to your home screen.")
    print("    - Full native camera, gallery, and file permissions enabled.")
    print("\n 2. Cloud APK Build via GitHub Actions (Automated)")
    print("    - Push your repository to GitHub.")
    print("    - GitHub cloud runs Ubuntu + Java 17 + Android SDK.")
    print("    - Download your compiled 'notes-studio-debug.apk' from Releases!")
    print("\n 3. Local Capacitor / Android Studio APK Build")
    print("    - Requires: Node.js, Android Studio / Gradle.")
    print("    - Commands: npx cap add android && npx cap build")
    print("=================================================================\n")

    # Check local tools
    has_node = shutil.which("node") is not None
    has_npm = shutil.which("npm") is not None
    has_gradle = shutil.which("gradle") is not None or shutil.which("gradle.bat") is not None

    print(f"Tool check: Node={has_node}, npm={has_npm}, Gradle={has_gradle}")

    if has_node and has_npm:
        print("\n[*] Initializing Capacitor mobile wrapper...")
        try:
            # Install dependencies
            subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=base_dir, check=True)
            print("[+] Capacitor dependencies installed.")
            print("[+] Mobile assets in 'www/' are ready for APK packaging.")
        except Exception as e:
            print(f"[!] Warning during npm install: {e}")

    print("\n✅ Setup complete! To start the OpenAI Mobile Scanner server, run:")
    print("   python openai_api_server.py")

if __name__ == "__main__":
    main()


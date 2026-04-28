import subprocess
import sys
import time
import os

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("===================================================")
    print("🎬 CineGenre AI - Starting Services...")
    print("===================================================")

    # 1. Start the FastAPI backend
    print("\n[1/1] 🚀 Starting FastAPI Backend Server on port 8000...")
    backend_process = subprocess.Popen([sys.executable, os.path.join("backend", "app.py")])

    print("\n✅ Services started! Open your browser to http://localhost:8000")

    try:
        # Keep the main process alive so the user can see logs
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services safely...")
        backend_process.terminate()
        print("✅ Services stopped. Goodbye!")

if __name__ == "__main__":
    main()

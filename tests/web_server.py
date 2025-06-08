import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brax_training_viewer.web_server import run_web_server

if __name__ == "__main__":
    # print("Launching Brax Training Viewer web server at http://127.0.0.1:8000 ...")
    run_web_server(host="127.0.0.1", port=8000, reload=False) 
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from braxviewer.WebViewer import WebViewer
from braxviewer.WebSocketStreamer import WebSocketStreamer

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    fps = 30

    viewer = WebViewer(host=host, port=port)
    viewer.run()

    streamer = WebSocketStreamer(uri=f"ws://{host}:{port}/ws/frame", fps=fps)
    streamer.start()

    while True:
        pass
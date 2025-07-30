"""A script to run the WebViewer server independently for the Cartpole example."""

import time
import sys
import os

# Add project path to allow for relative imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from braxviewer.WebViewer import WebViewer
import config

def main():
    """Starts the WebViewer server."""
    viewer = WebViewer(
        host=config.HOST,
        port=config.PORT,
        xml=config.XML_MODEL,
        num_envs=config.NUM_PARALLEL_ENVS
    )
    # Run the server in a thread so it doesn't block.
    viewer.run()

    print(f"Viewer server started. Visit http://{config.HOST}:{config.PORT}")
    print("Press Ctrl+C to stop.")

    try:
        # Keep the main thread alive to allow the server to run.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping viewer server...")
        viewer.stop()
        print("Server stopped.")

if __name__ == '__main__':
    main() 
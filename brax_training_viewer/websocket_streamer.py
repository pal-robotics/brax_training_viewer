import asyncio
import threading
import queue
import json as std_json
import time
import websockets

from brax_training_viewer.utils import state_to_dict


class WebSocketStateStreamer:
    """
    An object-oriented WebSocket streamer for Brax pipeline states.
    Sends states from a queue at a specified frames-per-second (fps) rate.
    """

    def __init__(self, uri="ws://localhost:8000/ws/frame", fps=30):
        self.uri = uri
        self.fps = fps
        self.frame_delay = 1.0 / fps
        self._state_queue = queue.Queue()
        self._thread = None
        self._started = False

    def start(self):
        """Starts the background thread that handles streaming."""
        if not self._started:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._started = True

    def _run(self):
        """Internal runner function for the thread."""
        asyncio.run(self._ws_loop())

    async def _ws_loop(self):
        """Async loop that sends states via WebSocket at fixed fps."""
        try:
            async with websockets.connect(self.uri) as ws:
                print(f"[WebSocket] Connected to {self.uri}")
                while True:
                    start_time = time.time()

                    # Blocking get in executor to avoid blocking event loop
                    state = await asyncio.get_event_loop().run_in_executor(None, self._state_queue.get)
                    if state is None:
                        break

                    try:
                        frame_json = std_json.dumps(state_to_dict(state))
                        await ws.send(frame_json)
                    except Exception as e:
                        print(f"[WebSocket] Failed to send frame: {e}")

                    # Wait remaining time to match fps
                    elapsed = time.time() - start_time
                    wait_time = self.frame_delay - elapsed
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
        except Exception as e:
            print(f"[WebSocket] Connection error: {e}")

    def send(self, state):
        """Send a state to the WebSocket stream."""
        self._state_queue.put(state)

    def stop(self):
        """Stops the WebSocket thread (safely)."""
        self._state_queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._started = False

import asyncio
import websockets
from threading import Thread
from queue import Queue

class WebSocketStreamer:
    def __init__(self, uri, serialize_frame):
        self.uri = uri
        self._serialize_frame = serialize_frame
        self._state_queue = Queue()
        self._started = False
        self._thread = None

    def start(self):
        """Starts the WebSocket thread."""
        self._started = True
        self._thread = Thread(target=self._run)
        self._thread.start()

    def _run(self):
        """Runs the asyncio event loop."""
        asyncio.run(self._ws_loop())

    async def _ws_loop(self):
        """Async loop that sends states via WebSocket with auto-reconnect."""
        loop = asyncio.get_event_loop()
        while self._started:
            try:
                async with websockets.connect(self.uri) as ws:
                    print("[WebSocket] Streamer connected.")
                    while True:
                        # Get an item from the synchronous queue in a non-blocking way.
                        item = await loop.run_in_executor(None, self._state_queue.get)
                        try:
                            frame_json = await loop.run_in_executor(
                                None, self._serialize_frame, item
                            )
                            await ws.send(frame_json)
                        except Exception as e:
                            print(f"[WebSocket] Failed to send frame: {e}")
                            continue
            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
                print(f"[WebSocket] Connection error ({e}), retrying in 3 seconds...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"[WebSocket] An unexpected error occurred: {e}")
                break

    def send(self, state, discard_queue: bool = False):
        """
        Sends a state to be sent via WebSocket.
        If discard_queue is True, discards all previous states in the queue.
        """
        if discard_queue:
            with self._state_queue.mutex:
                self._state_queue.queue.clear()
        self._state_queue.put(state)

    def stop(self):
        """Stops the WebSocket thread safely."""
        if self._started:
            self._started = False
            self._state_queue.put(None)  # Send sentinel to stop the loop
            if self._thread and self._thread.is_alive():
                self._thread.join()
            print("[WebSocket] Streamer stopped.")

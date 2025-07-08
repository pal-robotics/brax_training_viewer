import asyncio
import threading
import queue
import json as std_json
import time
import websockets

from braxviewer.utils import state_to_dict


class WebSocketStreamer:
    """
    An optimized WebSocket streamer for the frontend that offloads CPU-bound work
    and automatically reconnects.
    """

    def __init__(self, uri="ws://localhost:8000/ws/frame", unbatched: bool = True):
        self.uri = uri
        self._state_queue = queue.Queue()
        self._thread = None
        self._started = False
        self.unbatched = unbatched

    def start(self):
        """Starts the background thread that handles streaming."""
        if not self._started:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._started = True

    def _run(self):
        """Internal runner function for the thread."""
        asyncio.run(self._ws_loop())

    def _serialize_frame(self, state) -> str:
        """CPU-bound serialization task."""
        return std_json.dumps(state_to_dict(state, unbatched=self.unbatched))

    async def _ws_loop(self):
        """Async loop that sends states via WebSocket with auto-reconnect."""
        loop = asyncio.get_event_loop()
        while self._started:
            try:
                async with websockets.connect(self.uri) as ws:
                    # print("[WebSocket] Streamer connected.")
                    while True:
                        # Get a state from the synchronous queue in a non-blocking way.
                        state = await loop.run_in_executor(None, self._state_queue.get)
                        if state is None:  # Sentinel value to stop the loop.
                            return

                        try:
                            # Run the CPU-bound serialization in a thread pool executor.
                            frame_json = await loop.run_in_executor(
                                None, self._serialize_frame, state
                            )
                            await ws.send(frame_json)
                        except Exception as e:
                            # print(f"[WebSocket] Failed to send frame: {e}")
                            continue
            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
                # print(f"[WebSocket] Connection error ({e}), retrying in 3 seconds...")
                await asyncio.sleep(3)
            except Exception as e:
                # print(f"[WebSocket] An unexpected error occurred: {e}")
                break

    def send(self, state, discard_queue: bool = False):
        """
        Sends a state to the WebSocket stream.

        Args:
            state: The state to send.
            discard_queue: If True, discard all previous states in the queue
                           to only send the latest.
        """
        if discard_queue:
            # Efficiently clear the queue by consuming all existing items.
            while not self._state_queue.empty():
                try:
                    self._state_queue.get_nowait()
                except queue.Empty:
                    break
        self._state_queue.put(state)

    def stop(self):
        """Stops the WebSocket thread safely."""
        if self._started:
            self._started = False
            self._state_queue.put(None)  # Send sentinel to stop the loop
            if self._thread and self._thread.is_alive():
                self._thread.join()
            # print("[WebSocket] Streamer stopped.")
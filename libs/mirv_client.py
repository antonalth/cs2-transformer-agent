# mirv_client.py

import json
import threading
import websocket  # pip install websocket-client
import time

HOST = "localhost"
PORT = 31337
PATH = "mirv"
CLIENT_ID = 69
URI = f"ws://{HOST}:{PORT}/{PATH}?clientId={CLIENT_ID}"


class MirvClient:
    """
    Minimal MIRV WebSocket client.

    Usage:
        client = MirvClient()
        client.sendCommand("exec_this")
        ...
        client.close()
    """

    def __init__(self, uri: str = URI):
        self._uri = uri
        self._ws_app = websocket.WebSocketApp(
            self._uri,
            on_message=self._on_message,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(target=self._ws_app.run_forever)
        self._thread.start()
        time.sleep(5)
        self._connected = True

    def _on_message(self, ws, message: str):
        # Raw output from the server—feel free to replace with a JSON parser or logger.
        print(f"<< {message}")

    def _on_close(self, ws, code, reason):
        print(f"## Connection closed (code={code}) {reason or ''}")
        self._connected = False

    def sendCommand(self, cmd: str):
        """
        Send an "exec" event to the server with the given command string.
        """
        if not self._connected:
            raise RuntimeError("WebSocket is not connected or has already been closed.")
        payload = {
            "eventName": "exec",
            "values": [cmd],
        }
        self._ws_app.send(json.dumps(payload))

    def close(self):
        """
        Close the WebSocket gracefully and wait for the thread to exit.
        """
        if self._connected:
            self._ws_app.close()
            self._thread.join()
            self._connected = False


def connect() -> MirvClient:
    """
    Create a MirvClient, start its background thread, and return it.
    """
    return MirvClient()


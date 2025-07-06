import asyncio
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

# You must have the websockets library installed: pip install websockets
import websockets
from websockets.server import serve
from websockets.exceptions import ConnectionClosed

# --- Configuration ---
WS_HOST = "0.0.0.0"
WS_PORT = 31337
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8080

# --- Shared State ---
GAME_CLIENTS = {}
clients_lock = threading.Lock()

# --- WebSocket Server Logic (asyncio) ---

async def game_client_handler(websocket, path):
    """
    Handles connections from game clients. This is the final, stable version.
    """
    client_id = None
    
    if path == '/mirv' and 'hlae' in websocket.query_params:
        client_id = "hlae-main"
    elif path != '/':
        client_id = path.strip('/')

    if not client_id:
        print(f"[WS] Connection rejected from {websocket.remote_address}: Could not determine client ID from URL.")
        await websocket.close(1008, "Could not determine client ID from connection URL.")
        return

    with clients_lock:
        if client_id in GAME_CLIENTS:
            print(f"[WS] Connection rejected: ID '{client_id}' is already connected.")
            await websocket.close(1008, f"Client ID '{client_id}' is already in use.")
            return
        GAME_CLIENTS[client_id] = websocket
        print(f"[WS] ✅ Game client connected: '{client_id}'")
        print(f"[WS]    Total clients: {len(GAME_CLIENTS)}")

    try:
        # *** THE CRUCIAL FIX ***
        # We must actively listen for incoming messages to keep the connection healthy.
        # Even if we don't process the messages, consuming them from the
        # stream prevents buffers from filling up and causing a forced disconnect.
        async for message in websocket:
            # In this simple command server, we don't need to process messages
            # from the game client, but we MUST iterate through them.
            pass
    except ConnectionClosed:
        # This is the expected exception when the client disconnects gracefully.
        # We can just ignore it and proceed to the finally block.
        pass
    except Exception as e:
        # Log other potential errors during the connection.
        print(f"[WS] An unexpected error occurred with client '{client_id}': {e}")
    finally:
        # Safely unregister the client upon any kind of disconnection.
        with clients_lock:
            if client_id in GAME_CLIENTS:
                del GAME_CLIENTS[client_id]
            print(f"[WS] ❌ Game client disconnected: '{client_id}'")
            print(f"[WS]    Total clients: {len(GAME_CLIENTS)}")


# --- HTTP Server Logic (threading) ---
# This section remains unchanged and is correct.

class CommandHTTPHandler(BaseHTTPRequestHandler):
    """Handles incoming HTTP GET requests to control the game clients."""
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == '/listavailable':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            with clients_lock:
                client_ids = list(GAME_CLIENTS.keys())
            self.wfile.write(json.dumps(client_ids).encode('utf-8'))
            return

        if parsed_path.path == '/sendcommand':
            params = urllib.parse.parse_qs(parsed_path.query)
            target_id = params.get('id', [None])[0]
            command = params.get('command', [None])[0]

            if not target_id or not command:
                self.send_error(400, "Bad Request: 'id' and 'command' parameters are required.")
                return
            
            with clients_lock:
                target_socket = GAME_CLIENTS.get(target_id)
            
            if target_socket:
                payload = { "eventName": "exec", "values": [command] }
                message_to_send = json.dumps(payload)
                asyncio.run_coroutine_threadsafe(
                    target_socket.send(message_to_send), self.server.loop
                )
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": f"Command sent to '{target_id}'."}).encode('utf-8'))
                print(f"[HTTP] -> Relaying command to '{target_id}': {command}")
            else:
                self.send_error(404, f"Not Found: Client with id '{target_id}' is not connected.")
            return
            
        self.send_error(404, "Not Found: Use /listavailable or /sendcommand?id=...&command=...")

def start_http_server(loop):
    server_address = (HTTP_HOST, HTTP_PORT)
    httpd = HTTPServer(server_address, CommandHTTPHandler)
    httpd.loop = loop
    print(f"[HTTP] Listening for commands on http://{HTTP_HOST}:{HTTP_PORT}")
    httpd.serve_forever()


async def main():
    """Main function to start and manage both servers."""
    print("--- Full and Final Compatible Command Server ---")
    main_loop = asyncio.get_running_loop()
    http_thread = threading.Thread(target=start_http_server, args=(main_loop,), daemon=True)
    http_thread.start()

    async with serve(
        game_client_handler,
        WS_HOST,
        WS_PORT,
        ping_interval=20,
        ping_timeout=20,
    ):
        print(f"[WS] Listening for game clients on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n--- Server shutting down ---")
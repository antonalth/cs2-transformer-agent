import asyncio
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from websockets.server import serve
from websockets.exceptions import ConnectionClosed

# --- Configuration ---
WS_HOST = "0.0.0.0"
WS_PORT = 31337
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8080

# --- Shared State ---
# This dictionary stores active game client connections.
# Key: client_id, Value: websocket connection object.
GAME_CLIENTS = {}
# A lock is crucial for thread-safe access to the shared GAME_CLIENTS dict.
clients_lock = threading.Lock()

# --- WebSocket Server Logic (asyncio) ---

async def game_client_handler(websocket, path):
    """
    Handles connections from game clients.
    It determines the client ID from the connection URL, supporting both the
    original "?hlae=1" format and a future path-based ID.
    """
    client_id = None
    parsed_uri = urllib.parse.urlparse(websocket.request_headers['uri'])
    
    # Check for the original "?hlae=1" format
    if parsed_uri.path == '/mirv':
        query_params = urllib.parse.parse_qs(parsed_uri.query)
        if query_params.get('hlae'):
            client_id = "hlae-main" # Assign a default ID for the main HLAE instance
    else:
        # Support for future path-based IDs (e.g., /my-game-id)
        client_id = parsed_uri.path.strip('/')

    if not client_id:
        print(f"[WS] Connection rejected from {websocket.remote_address}: Could not determine client ID from URI.")
        await websocket.close(1008, "Could not determine client ID.")
        return

    # Safely register the new client
    with clients_lock:
        if client_id in GAME_CLIENTS:
            print(f"[WS] Connection rejected: ID '{client_id}' is already connected.")
            await websocket.close(1008, f"Client ID '{client_id}' is already in use.")
            return
        GAME_CLIENTS[client_id] = websocket
        print(f"[WS] ✅ Game client connected: '{client_id}'")
        print(f"[WS]    Total clients: {len(GAME_CLIENTS)}")

    try:
        # We don't expect messages from the game client in this simple setup.
        # Just wait for it to disconnect.
        await websocket.wait_closed()
    finally:
        # Safely unregister the client
        with clients_lock:
            if client_id in GAME_CLIENTS:
                del GAME_CLIENTS[client_id]
            print(f"[WS] ❌ Game client disconnected: '{client_id}'")
            print(f"[WS]    Total clients: {len(GAME_CLIENTS)}")

# --- HTTP Server Logic (threading) ---

class CommandHTTPHandler(BaseHTTPRequestHandler):
    """Handles incoming HTTP GET requests to control the game clients."""

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        # --- Endpoint: /listavailable ---
        if parsed_path.path == '/listavailable':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            with clients_lock:
                client_ids = list(GAME_CLIENTS.keys())
            self.wfile.write(json.dumps(client_ids).encode('utf-8'))
            return

        # --- Endpoint: /sendcommand ---
        if parsed_path.path == '/sendcommand':
            params = urllib.parse.parse_qs(parsed_path.query)
            target_id = params.get('id', [None])[0]
            command = params.get('command', [None])[0]

            if not target_id or not command:
                self.send_error(400, "Bad Request: 'id' and 'command' parameters are required.")
                return
            
            # Find the target socket connection
            with clients_lock:
                target_socket = GAME_CLIENTS.get(target_id)
            
            if target_socket:
                # *** CRUCIAL CHANGE FOR COMPATIBILITY ***
                # We must wrap the command in the JSON format expected by simple-websockets.
                payload = {
                    "eventName": "exec",
                    "values": [command]
                }
                message_to_send = json.dumps(payload)
                
                # Schedule the async send operation on the main event loop
                asyncio.run_coroutine_threadsafe(
                    target_socket.send(message_to_send),
                    self.server.loop
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
    """Sets up and runs the HTTP server in its own thread."""
    server_address = (HTTP_HOST, HTTP_PORT)
    httpd = HTTPServer(server_address, CommandHTTPHandler)
    httpd.loop = loop # Give it a reference to the main asyncio loop
    print(f"[HTTP] Listening for commands on http://{HTTP_HOST}:{HTTP_PORT}")
    httpd.serve_forever()

async def main():
    """Starts both the WebSocket and HTTP servers."""
    print("--- Compatible Command Server ---")
    main_loop = asyncio.get_running_loop()

    # Start the HTTP server in a separate daemon thread
    http_thread = threading.Thread(target=start_http_server, args=(main_loop,), daemon=True)
    http_thread.start()

    # Start the WebSocket server
    async with serve(game_client_handler, WS_HOST, WS_PORT):
        print(f"[WS] Listening for game clients on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n--- Server shutting down ---")
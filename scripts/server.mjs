import http from 'http';
import { URL } from 'url';
import { SimpleWebSocketServer } from 'simple-websockets/server';

// --- Configuration ---
const HOST = '0.0.0.0'; // Listen on all network interfaces
const WS_PORT = 31337;
const HTTP_PORT = 8080;
const WS_PATH = '/mirv';

// --- Shared State ---
// This variable will hold the single, active HLAE socket connection.
let hlaeSocket = null;

/**
 * This function sets up the WebSocket server to listen for the HLAE client.
 */
function startWebSocketServer() {
    const wsHttpServer = http.createServer();

    const wss = new SimpleWebSocketServer({
        server: wsHttpServer,
        path: WS_PATH,
    });

    // Handle new WebSocket connections
    wss.onConnection((socket, req) => {
        const params = new URL(req.url, `ws://${req.headers.host}`).searchParams;

        // Check if this is the HLAE client based on the URL parameter
        if (params.has('hlae')) {
            if (hlaeSocket && hlaeSocket.isConnected()) {
                console.log('[WS] Rejecting new HLAE connection: An instance is already active.');
                // 1013: Try again later (service unavailable)
                socket._socket.close(1013, 'An HLAE instance is already connected.');
                return;
            }

            console.log('[WS] ✅ HLAE client has connected.');
            hlaeSocket = socket;

            // Handle disconnection
            socket.on('disconnect', () => {
                console.log('[WS] ❌ HLAE client has disconnected.');
                hlaeSocket = null;
            });

            // Handle errors
            socket.on('error', (err) => {
                console.error('[WS] HLAE socket error:', err.message);
            });
            
            // *** CRUCIAL FIX for WeakContextWrapper ERROR ***
            // We MUST actively listen for events coming FROM HLAE. This proves to the
            // client that we are a valid, functioning broker, preventing it from
            // terminating the connection, which would cause the JS context in HLAE to crash.
            // We don't have to DO anything with the data, just acknowledge we can receive it.
            const eventsFromHlae = ['onGameEvent', 'onCViewRenderSetupView', 'onAddEntity', 'onRemoveEntity'];
            for (const eventName of eventsFromHlae) {
                socket.on(eventName, () => {
                    // Discard the data, the listener just needs to exist.
                });
            }

            return;
        }

        // If it's not an HLAE client, we don't know what it is.
        console.log(`[WS] Rejecting unknown connection from path: ${req.url}`);
        socket._socket.close(1002, 'This server only accepts an HLAE client.');
    });

    wsHttpServer.listen(WS_PORT, HOST, () => {
        console.log(`[WS] Listening for HLAE client on ws://${HOST}:${WS_PORT}${WS_PATH}`);
    });
}

/**
 * This function sets up the HTTP server to listen for commands.
 */
function startHttpCommandServer() {
    const commandServer = http.createServer((req, res) => {
        const requestUrl = new URL(req.url, `http://${req.headers.host}`);
        const { pathname, searchParams } = requestUrl;

        // --- Endpoint: /listavailable ---
        if (pathname === '/listavailable') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            const clientIds = [];
            if (hlaeSocket && hlaeSocket.isConnected()) {
                clientIds.push('hlae-main'); // The static ID for our single HLAE instance
            }
            res.end(JSON.stringify(clientIds));
            return;
        }

        // --- Endpoint: /sendcommand ---
        if (pathname === '/sendcommand') {
            const targetId = searchParams.get('id');
            const command = searchParams.get('command');

            if (!targetId || !command) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: "Bad Request: 'id' and 'command' parameters are required." }));
                return;
            }

            if (targetId !== 'hlae-main') {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: `Not Found: Client with id '${targetId}' not found.` }));
                return;
            }

            if (!hlaeSocket || !hlaeSocket.isConnected()) {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: "Not Found: The HLAE client is not connected." }));
                return;
            }

            // Use the simple-websockets API to send the command in the correct format.
            // It automatically wraps it as { eventName: 'exec', values: [command] }
            hlaeSocket.send('exec', command);
            
            console.log(`[HTTP] -> Relaying command to HLAE: ${command}`);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ status: 'success', message: 'Command sent to HLAE.' }));
            return;
        }

        // --- Fallback for unknown paths ---
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not Found: Use /listavailable or /sendcommand' }));
    });

    commandServer.listen(HTTP_PORT, HOST, () => {
        console.log(`[HTTP] Listening for commands on http://${HOST}:${HTTP_PORT}`);
    });
}

/**
 * Main entry point to start all services.
 */
function main() {
    console.log('--- Node.js Command Server Starting ---');
    startWebSocketServer();
    startHttpCommandServer();
}

main();
// src/server.ts

import http from 'http';
import { URL } from 'url';
// CORRECTED IMPORTS: SimpleWebSocketServer is in a separate sub-module.
import { SimpleWebSocket } from 'simple-websockets';
import { SimpleWebSocketServer } from 'simple-websockets/server';

// --- Configuration ---
const WS_PORT = 31337;
const HTTP_PORT = 8080;
const HOST = '0.0.0.0'; // Listen on all network interfaces

// --- Shared State ---
// We will store all connected game clients in this Map.
// The key is the client's unique ID, and the value is the socket object.
const gameClients = new Map<string, SimpleWebSocket<any>>();

/**
 * Sets up and starts the WebSocket server to listen for game clients.
 */
function startWebSocketServer() {
    const wsHttpServer = http.createServer();

    const wss = new SimpleWebSocketServer({
        server: wsHttpServer,
    });

    // This event fires for every new WebSocket connection.
    wss.onConnection((socket, req) => {
        // The client ID is determined by the URL path (e.g., /instance-1 -> "instance-1").
        const clientId = (req.url || '/').slice(1);

        // --- Validation ---
        if (!clientId) {
            console.log('[WS] Rejecting connection: No client ID provided in path.');
            socket._socket.close(1008, 'A unique client ID is required in the URL path.');
            return;
        }

        if (gameClients.has(clientId)) {
            console.log(`[WS] Rejecting connection: Client ID '${clientId}' is already in use.`);
            socket._socket.close(1013, `Client ID '${clientId}' is already connected.`);
            return;
        }

        // --- Registration ---
        console.log(`[WS] ✅ Game client connected: '${clientId}'`);
        gameClients.set(clientId, socket);

        // --- Event Handling for this specific socket ---
        socket.on('disconnect', () => {
            console.log(`[WS] ❌ Game client disconnected: '${clientId}'`);
            gameClients.delete(clientId);
        });

        socket.on('error', (err) => {
            console.error(`[WS] Error on client '${clientId}':`, err.message);
        });

        // *** CRUCIAL STABILITY FIX ***
        // We must add listeners for events coming FROM the game client.
        // This proves to the client that we are a valid broker and prevents
        // the connection from being terminated, which causes the WeakContextWrapper crash.
        const eventsToKeepAlive = ['onGameEvent', 'onCViewRenderSetupView', 'onAddEntity', 'onRemoveEntity'];
        for (const eventName of eventsToKeepAlive) {
            socket.on(eventName, () => { /* Listening is all that matters. */ });
        }
    });

    wsHttpServer.listen(WS_PORT, HOST, () => {
        console.log(`[WS] Listening for game clients on ws://${HOST}:${WS_PORT}`);
    });
}

/**
 * Sets up and starts the HTTP server to listen for external commands.
 */
function startHttpCommandServer() {
    const commandServer = http.createServer((req, res) => {
        const requestUrl = new URL(req.url || '/', `http://${req.headers.host}`);
        const { pathname, searchParams } = requestUrl;

        // --- Endpoint: /listavailable ---
        if (pathname === '/listavailable') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            const clientIds = Array.from(gameClients.keys());
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

            const targetSocket = gameClients.get(targetId);

            if (!targetSocket || !targetSocket.isConnected()) {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: `Not Found: Client with id '${targetId}' is not connected.` }));
                return;
            }

            // The 'simple-websockets' library handles the JSON formatting for us.
            targetSocket.send('exec', command);
            
            console.log(`[HTTP] -> Relaying command to '${targetId}': ${command}`);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ status: 'success', message: `Command sent to '${targetId}'.` }));
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
    console.log('--- TypeScript Command Server Starting ---');
    startWebSocketServer();
    startHttpCommandServer();
}

main();
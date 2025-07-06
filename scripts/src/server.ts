// src/server.ts

import http from 'http';
import { URL } from 'url';
import { SimpleWebSocket } from 'simple-websockets';
import { SimpleWebSocketServer } from 'simple-websockets/server';

// --- Configuration ---
const WS_PORT = 31337;
const HTTP_PORT = 8080;
const HOST = '0.0.0.0'; // Listen on all network interfaces

// --- Shared State ---
const gameClients = new Map<string, SimpleWebSocket<any>>();

/**
 * Sets up and starts the WebSocket server to listen for game clients.
 */
function startWebSocketServer() {
    const wsHttpServer = http.createServer();

    const wss = new SimpleWebSocketServer({
        server: wsHttpServer,
    });

    wss.onConnection((socket, req) => {
        // *** THE FIX IS HERE: We now properly parse the URL to assign a clean ID ***
        const connectionUrl = new URL(req.url || '/', `ws://${req.headers.host}`);
        let clientId: string | null = null;

        // Case 1: Handle the original HLAE connection style
        if (connectionUrl.pathname === '/mirv' && connectionUrl.searchParams.has('hlae')) {
            clientId = 'hlae-main'; // Assign a clean, predictable, canonical ID
        } 
        // Case 2: Handle the new multi-client style where the ID is the path
        else if (connectionUrl.pathname !== '/') {
            clientId = connectionUrl.pathname.slice(1); // e.g., /instance-1 -> "instance-1"
        }

        // --- Validation (now uses the clean clientId) ---
        if (!clientId) {
            console.log('[WS] Rejecting connection: Could not determine a valid client ID from URL.');
            socket._socket.close(1008, 'A unique client ID is required in the URL path.');
            return;
        }

        if (gameClients.has(clientId)) {
            console.log(`[WS] Rejecting connection: Client ID '${clientId}' is already in use.`);
            socket._socket.close(1013, `Client ID '${clientId}' is already connected.`);
            return;
        }

        // --- Registration (uses the clean clientId) ---
        console.log(`[WS] ✅ Game client connected with clean ID: '${clientId}'`);
        gameClients.set(clientId, socket);

        // --- Event Handling for this specific socket ---
        socket.on('disconnect', () => {
            console.log(`[WS] ❌ Game client disconnected: '${clientId}'`);
            gameClients.delete(clientId);
        });

        socket.on('error', (err) => {
            console.error(`[WS] Error on client '${clientId}':`, err.message);
        });

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
 * This function is already correct and needs no changes.
 */
function startHttpCommandServer() {
    const commandServer = http.createServer((req, res) => {
        const requestUrl = new URL(req.url || '/', `http://${req.headers.host}`);
        const { pathname, searchParams } = requestUrl;

        if (pathname === '/listavailable') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            const clientIds = Array.from(gameClients.keys());
            res.end(JSON.stringify(clientIds));
            return;
        }

        if (pathname === '/sendcommand') {
            const targetId = searchParams.get('id');
            const command = searchParams.get('command');

            if (!targetId || !command) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: "Bad Request: 'id' and 'command' parameters are required." }));
                return;
            }

            const targetSocket = gameClients.get(targetId);

            if (!targetSocket || !targetSocket.connected) {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: `Not Found: Client with id '${targetId}' is not connected.` }));
                return;
            }

            targetSocket.send('exec', command);
            
            console.log(`[HTTP] -> Relaying command to '${targetId}': ${command}`);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ status: 'success', message: `Command sent to '${targetId}'.` }));
            return;
        }

        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not Found: Use /listavailable or /sendcommand' }));
    });

    commandServer.listen(HTTP_PORT, HOST, () => {
        console.log(`[HTTP] Listening for commands on http://${HOST}:${HTTP_PORT}`);
    });
}

function main() {
    console.log('--- TypeScript Command Server Starting ---');
    startWebSocketServer();
    startHttpCommandServer();
}

main();
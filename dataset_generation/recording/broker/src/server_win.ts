// This script acts as a bridge between multiple game clients and external applications.
// Game clients connect via WebSocket. Applications can interact with them via an HTTP API.
//
// Functionality:
// 1. WebSocket Server (Port 31337):
//    - Listens for incoming connections from game clients.
//    - Assigns a unique, incrementing ID to each client upon connection.
//    - Manages a list of active clients.
//
// 2. HTTP API Server (Port 8080):
//    - Provides a '/list' endpoint to get the IDs of all connected clients.
//    - Provides a '/run' endpoint to send a console command to a specific client by its ID.
//    - Provides a '/running' endpoint to check if a specific process is running in a client's sandbox.
//
// We use the 'simple-websockets' library for convenience.
// It sends events in the following format: { eventName: string, values: unknown[] }
// To execute a command, we send an 'exec' event.

import { execFile } from 'child_process';
import { promisify } from 'util';
import http from 'http';
import { SimpleWebSocketServer } from 'simple-websockets/server';
import { URL } from 'url';

// Define the event map for our simple API. We only need to send 'exec' commands.
interface ApiEventMap {
	exec: (command: string) => void;
}

const API_EVENT_MAP: ApiEventMap = {
	exec: (command:string) => {},
};

// Promisify execFile. It is more secure and handles arguments better than exec.
const execFilePromise = promisify(execFile);

// --- Configuration ---
const WEBSOCKET_HOST = 'localhost';
const WEBSOCKET_PORT = 31337;
const HTTP_API_PORT = 8080;

// --- State Management ---
// A map to store connected game clients, with our assigned ID as the key.
const gameClients = new Map();
let nextClientId = 1; // Used to generate unique IDs for clients.

// --- WebSocket Server for Game Clients ---
function createWebSocketServer() {
	const wsServer = http.createServer();
	const wss = new SimpleWebSocketServer({
		server: wsServer,
	});

	wss.onConnection((socket, req) => {
		// Assign a unique ID to the new client.
		const clientId = (nextClientId++).toString();
		gameClients.set(clientId, socket);
		console.log(`[WebSocket] Game client connected. Assigned ID: ${clientId}`);

		// Handle client disconnection.
		socket.on('disconnect', () => {
			gameClients.delete(clientId);
			console.log(`[WebSocket] Game client ${clientId} has disconnected.`);
		});

        // We don't need to listen for any events from the client in this setup.
	});

	wsServer.listen(WEBSOCKET_PORT, WEBSOCKET_HOST, () => {
		console.log(`[WebSocket] Server listening for game clients on ws://${WEBSOCKET_HOST}:${WEBSOCKET_PORT}`);
	});
}

// --- HTTP API Server for Applications ---
function createHttpApiServer() {
	const apiServer = http.createServer(async (req, res) => {
        // Use the URL constructor for robust parsing of path and query parameters.
		const requestUrl = new URL(req.url ?? '/', `http://${req.headers.host}`);
		
		// Endpoint: /list
		// Returns a JSON array of connected client IDs.
		if (requestUrl.pathname === '/list') {
			const clientIds = Array.from(gameClients.keys());
			res.writeHead(200, { 'Content-Type': 'application/json' });
			res.end(JSON.stringify(clientIds));
			return;
		}

		// Endpoint: /run?id=...&cmd=...
		// Sends a command to a specific client.
		if (requestUrl.pathname === '/run') {
			const clientId = requestUrl.searchParams.get('id');
			const commandToRun = requestUrl.searchParams.get('cmd');

			if (!clientId || !commandToRun) {
				res.writeHead(400, { 'Content-Type': 'text/plain' });
				res.end('Bad Request: Missing "id" or "cmd" query parameter.');
				return;
			}

			const clientSocket = gameClients.get(clientId);

			if (!clientSocket) {
				res.writeHead(404, { 'Content-Type': 'text/plain' });
				res.end(`Not Found: No client with ID "${clientId}" is connected.`);
				return;
			}

			// Send the command to the game client using the 'exec' event.
			clientSocket.send('exec', commandToRun);

			res.writeHead(200, { 'Content-Type': 'text/plain' });
			res.end(`Command sent to client ${clientId}.`);
			console.log(`[HTTP API] Sent command to client ${clientId}: ${commandToRun}`);
			return;
		}

		// Endpoint: /running?id=X&name=processname.exe
		// Checks if a given process name is running inside the specified client's sandbox.
		if (requestUrl.pathname === '/running') {
			const clientId = requestUrl.searchParams.get('id');
			const processName = requestUrl.searchParams.get('name');

			if (!clientId || !processName) {
				res.writeHead(400, { 'Content-Type': 'text/plain' });
				res.end('Bad Request: Missing "id" or "name" query parameter.');
				return;
			}

			const sandboxName = `game${clientId}`;
			
			// Set the Current Working Directory (cwd) to avoid path quoting issues.
			// FIX: Changed the WSL path to a native Windows path.
			const sandboxieDir = 'C:/Program Files/Sandboxie-Plus/';
			const execOptions = { cwd: sandboxieDir };

			// Now the command is simple, as we're already in the right directory.
			const listPidsCmdString = `Start.exe /box:${sandboxName} /listpids`;
        try {
				// --- Step 1: Get all PIDs running inside the specific sandbox. (First external call) ---
				const { stdout: pidsOutput } = await execFilePromise('cmd.exe', ['/c', listPidsCmdString], execOptions);
				const sandboxedPids = new Set(pidsOutput.match(/\d+/g) || []);

				if (sandboxedPids.size === 0) {
					// No processes in the sandbox, so it can't be running.
					res.writeHead(200, { 'Content-Type': 'application/json' });
					res.end('false');
					return;
				}

				// --- Step 2: Get all running processes on the system. (Second and final external call) ---
				// We use the native 'tasklist' command, which is very fast.
				// '/fo csv' formats the output as CSV. '/nh' removes the header row.
				const { stdout: tasklistOutput } = await execFilePromise('tasklist', ['/fo', 'csv', '/nh']);

				let isRunning = false;
				const targetProcessName = processName.toLowerCase();

				// --- Step 3: Parse the tasklist output and check for a match in memory. ---
				// This is extremely fast as it doesn't involve any more external calls.
				const runningProcesses = tasklistOutput.trim().split('\n');
				for (const processLine of runningProcesses) {
					// tasklist CSV format is: "Image Name","PID","Session Name",...
					// We only need the first two values.
					const parts = processLine.split('","');
					if (parts.length < 2) continue;

					const currentProcessName = parts[0].substring(1).toLowerCase(); // Remove leading quote
					const currentPid = parts[1];
					
					// If the process name matches AND its PID is in our sandbox set, we have a winner.
					if (currentProcessName === targetProcessName && sandboxedPids.has(currentPid)) {
						isRunning = true;
						break; // Found it, no need to check further.
					}
				}

				res.writeHead(200, { 'Content-Type': 'application/json' });
				res.end(JSON.stringify(isRunning));

			} catch (error) {
				console.error(`[HTTP API /running] Error checking process for client ${clientId}:`, error);
				// If something went wrong (e.g., sandbox doesn't exist), report 'false'.
				res.writeHead(200, { 'Content-Type': 'application/json' });
				res.end('false');
			}
			return;
		}

		// Handle unknown paths
		res.writeHead(404, { 'Content-Type': 'text/plain' });
		res.end('Not Found');
	});

	apiServer.listen(HTTP_API_PORT, () => {
		console.log(`[HTTP API] Server listening on http://localhost:${HTTP_API_PORT}`);
		console.log(`  > Usage: /list`);
		console.log(`  > Usage: /run?id=<CLIENT_ID>&cmd=<URL_ENCODED_COMMAND>`);
		console.log(`  > Usage: /running?id=<CLIENT_ID>&name=<PROCESS_NAME.EXE>`);
	});
}

// --- Main Execution ---
function main() {
	createWebSocketServer();
	createHttpApiServer();
}

main();
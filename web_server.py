"""FastAPI web server for the WhatsApp-style Web UI.

Provides:
  GET  /        -- serves the single-page HTML app
  WS   /ws      -- streams agent events to the browser in real time
  POST /start   -- launches the agent in a background thread
  POST /stop    -- cancels the running agent
"""

import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from display import WebDisplay

app = FastAPI(title="RedAgent Web UI")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_display: WebDisplay | None = None
_agent_thread: threading.Thread | None = None
_agent_stop_event = threading.Event()

# Each connected WebSocket gets its own asyncio.Queue for broadcast delivery.
_client_queues: dict[WebSocket, asyncio.Queue] = {}
_client_queues_lock = asyncio.Lock()

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Broadcast helper — called by WebDisplay._put() via the event loop
# ---------------------------------------------------------------------------

def _broadcast_event(event: dict):
    """Put *event* onto every connected client's queue (non-blocking)."""
    for q in _client_queues.values():
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # drop if a client is too slow


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """Serve the single-page chat UI."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.post("/start")
async def start_agent(max_iterations: int = 1000):
    """Kick off the agent in a background thread."""
    global _display, _agent_thread, _agent_stop_event

    if _agent_thread is not None and _agent_thread.is_alive():
        return JSONResponse({"status": "already_running"}, status_code=409)

    _agent_stop_event.clear()
    _display = WebDisplay(broadcast_fn=_broadcast_event)

    def _run():
        try:
            from graph import AgentGraph  # lazy import (heavy deps)
            agent = AgentGraph(display=_display)
            agent.run(max_iterations=max_iterations)
        except Exception as exc:
            _display.status(f"Agent error: {exc}")
            _display.finished(success=False)

    _agent_thread = threading.Thread(target=_run, daemon=True)
    _agent_thread.start()
    return {"status": "started"}


@app.post("/stop")
async def stop_agent():
    """Request the agent to stop (best-effort -- sets the stop flag)."""
    global _agent_thread
    _agent_stop_event.set()
    if _agent_thread is not None:
        _agent_thread = None
    if _display is not None:
        _display.finished(success=False)
    return {"status": "stopped"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Stream display events to the browser as JSON messages."""
    await ws.accept()

    # Create a per-client queue for this connection
    client_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    _client_queues[ws] = client_queue

    try:
        # Send a welcome status so the client knows it's connected
        await ws.send_json({
            "type": "status",
            "message": "Connected to RedAgent server. Press Start to begin.",
            "timestamp": "",
        })

        # Two concurrent tasks:
        #  1. Drain this client's queue and forward events
        #  2. Listen for incoming messages (keepalive / control)
        await asyncio.gather(
            _drain_client_queue(ws, client_queue),
            _listen(ws),
        )
    except WebSocketDisconnect:
        pass
    finally:
        _client_queues.pop(ws, None)


async def _drain_client_queue(ws: WebSocket, queue: asyncio.Queue):
    """Forward events from this client's personal queue to its WebSocket."""
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
            await ws.send_json(event)
        except asyncio.TimeoutError:
            continue
        except Exception:
            break


async def _listen(ws: WebSocket):
    """Receive messages from the browser (keepalive / future commands)."""
    try:
        while True:
            data = await ws.receive_text()
            # Currently unused; could handle pause/resume later
    except WebSocketDisconnect:
        pass

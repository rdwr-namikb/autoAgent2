import sys
import argparse
import webbrowser
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="RedAgent - Behavioral Learning Agent")
    parser.add_argument(
        "--mode", choices=["cli", "web"], default="cli",
        help="Display mode: 'cli' for terminal chat, 'web' for browser UI (default: cli)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port for the web UI server (default: 8080)",
    )
    parser.add_argument(
        "max_iterations", nargs="?", type=int, default=1000,
        help="Maximum number of iterations (default: 1000)",
    )
    args = parser.parse_args()

    if args.mode == "web":
        _run_web(args.port)
    else:
        _run_cli(args.max_iterations)


def _run_cli(max_iterations: int):
    """Run the agent with the classic CLI display."""
    from graph import AgentGraph
    from display import CLIDisplay

    display = CLIDisplay()
    agent = AgentGraph(display=display)
    agent.run(max_iterations=max_iterations)


def _run_web(port: int):
    """Launch the FastAPI web UI and open the browser."""
    import uvicorn
    from web_server import app

    url = f"http://localhost:{port}"
    print(f"Starting RedAgent Web UI at {url}")
    print("Press Ctrl+C to stop.\n")

    # Open the browser after a short delay so the server is ready
    import threading
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()

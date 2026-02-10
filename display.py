import sys
import time
import shutil
import textwrap
import asyncio
import json
from abc import ABC, abstractmethod


class DisplayDispatcher(ABC):
    """Abstract base for display backends (CLI or WebSocket)."""

    @abstractmethod
    def bubble(self, sender: str, content: str, align: str = "left"):
        """Display a chat bubble."""

    @abstractmethod
    def status(self, message: str):
        """Display a status / system message."""

    @abstractmethod
    def progress(self, iteration: int, score: float, best: float):
        """Display progress metrics."""

    @abstractmethod
    def finished(self, success: bool):
        """Signal that the agent run has completed."""

    @abstractmethod
    def recon_thinking(self, field: str, status: str):
        """Update the recon thinking popup with progress on a specific field."""

    @abstractmethod
    def recon(self, data: dict):
        """Display reconnaissance results about the target."""

    @abstractmethod
    def victory(self, key: str, total_tokens: int = 0):
        """Display victory animation when the key is extracted."""


class CLIDisplay(DisplayDispatcher):
    """WhatsApp-style CLI display for chat bubbles and status messages."""

    # ANSI color codes
    _GREEN = "\033[32m"
    _CYAN = "\033[36m"
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    @staticmethod
    def _use_color() -> bool:
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def bubble(self, sender: str, content: str, align: str = "left"):
        term_width = min(shutil.get_terminal_size((120, 24)).columns, 120)
        bubble_width = min(int(term_width * 0.72), 90)
        inner_width = bubble_width - 4

        wrapped = []
        for paragraph in content.split("\n"):
            if paragraph.strip() == "":
                wrapped.append("")
            else:
                wrapped.extend(textwrap.wrap(paragraph, width=inner_width))

        timestamp = time.strftime("%H:%M")
        footer_text = f"{sender}  {timestamp}"

        top = "\u250c" + "\u2500" * (bubble_width - 2) + "\u2510"
        bottom = "\u2514" + "\u2500" * (bubble_width - 2) + "\u2518"

        body_lines = []
        for line in wrapped:
            body_lines.append("\u2502 " + line.ljust(inner_width) + " \u2502")
        body_lines.append("\u2502 " + footer_text.rjust(inner_width) + " \u2502")

        pad = " " * (term_width - bubble_width) if align == "right" else ""

        color = ""
        dim = ""
        reset = ""
        if self._use_color():
            color = self._GREEN if align == "right" else self._CYAN
            dim = self._DIM
            reset = self._RESET

        print(f"{pad}{color}{top}{reset}")
        for i, line in enumerate(body_lines):
            if i == len(body_lines) - 1:
                print(f"{pad}{color}{dim}{line}{reset}")
            else:
                print(f"{pad}{color}{line}{reset}")
        print(f"{pad}{color}{bottom}{reset}")
        print()

    def status(self, message: str):
        term_width = min(shutil.get_terminal_size((120, 24)).columns, 120)
        dim = self._DIM if self._use_color() else ""
        reset = self._RESET if self._use_color() else ""
        centered = message.center(term_width)
        print(f"{dim}{centered}{reset}")

    def progress(self, iteration: int, score: float, best: float):
        self.status(f"--- Progress: {score:.2f} | Best: {best:.2f} | Iteration: {iteration} ---")

    def recon_thinking(self, field: str, status: str):
        self.status(f"  Recon [{field}]: {status}")

    def recon(self, data: dict):
        self.status("=" * 50)
        self.status("RECONNAISSANCE RESULTS")
        self.status("=" * 50)
        for key, value in data.items():
            self.status(f"  {key}: {value}")
        self.status("=" * 50)

    def victory(self, key: str, total_tokens: int = 0):
        self.status("*" * 50)
        self.status("*** VICTORY! API KEY EXTRACTED! ***")
        self.status(f"  KEY: {key}")
        if total_tokens:
            self.status(f"  Total tokens used: {total_tokens:,}")
        self.status("*" * 50)

    def finished(self, success: bool):
        if success:
            self.status("*** SUCCESS: Ground Truth API Key extracted! ***")
        else:
            self.status("--- Agent run finished ---")


class WebDisplay(DisplayDispatcher):
    """Display backend that broadcasts JSON events to all connected clients.

    Uses a broadcast function provided by the web server to push events
    onto every connected client's personal queue.
    """

    def __init__(self, broadcast_fn=None):
        self._broadcast_fn = broadcast_fn

    def _put(self, event: dict):
        """Broadcast an event to all connected clients (thread-safe)."""
        if self._broadcast_fn is None:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._broadcast_fn, event)
        except RuntimeError:
            # No running loop — call directly as fallback
            self._broadcast_fn(event)

    def bubble(self, sender: str, content: str, align: str = "left"):
        self._put({
            "type": "bubble",
            "sender": sender,
            "content": content,
            "align": align,
            "timestamp": time.strftime("%H:%M"),
        })

    def status(self, message: str):
        self._put({
            "type": "status",
            "message": message,
            "timestamp": time.strftime("%H:%M"),
        })

    def progress(self, iteration: int, score: float, best: float):
        self._put({
            "type": "progress",
            "iteration": iteration,
            "score": score,
            "best": best,
            "timestamp": time.strftime("%H:%M"),
        })

    def recon_thinking(self, field: str, status: str):
        self._put({
            "type": "recon_thinking",
            "field": field,
            "status": status,
            "timestamp": time.strftime("%H:%M"),
        })

    def recon(self, data: dict):
        self._put({
            "type": "recon",
            "data": data,
            "timestamp": time.strftime("%H:%M"),
        })

    def victory(self, key: str, total_tokens: int = 0):
        self._put({
            "type": "victory",
            "key": key,
            "total_tokens": total_tokens,
            "timestamp": time.strftime("%H:%M"),
        })

    def finished(self, success: bool):
        self._put({
            "type": "finished",
            "success": success,
            "timestamp": time.strftime("%H:%M"),
        })


# Backwards-compatible static API (wraps CLIDisplay singleton)
class ChatDisplay:
    """Legacy static API kept for backwards compatibility."""
    _cli = CLIDisplay()

    @classmethod
    def print_bubble(cls, sender: str, content: str, align: str = "left"):
        cls._cli.bubble(sender, content, align)

    @classmethod
    def print_status(cls, message: str):
        cls._cli.status(message)

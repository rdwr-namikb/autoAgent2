import sys
import time
import shutil
import textwrap


class ChatDisplay:
    """WhatsApp-style CLI display for chat bubbles and status messages."""

    # ANSI color codes
    _GREEN = "\033[32m"
    _CYAN = "\033[36m"
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    @staticmethod
    def _use_color() -> bool:
        """Check if stdout is a real terminal (supports ANSI colors)."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    @classmethod
    def print_bubble(cls, sender: str, content: str, align: str = "left"):
        """Print a WhatsApp-style chat bubble.

        Args:
            sender:  Label shown in the footer (e.g. 'You', 'Target').
            content: Message body text.
            align:   'right' for sent messages, 'left' for received messages.
        """
        term_width = min(shutil.get_terminal_size((120, 24)).columns, 120)
        bubble_width = min(int(term_width * 0.72), 90)
        inner_width = bubble_width - 4  # "| " + " |"

        # Wrap content into lines that fit the bubble
        wrapped = []
        for paragraph in content.split("\n"):
            if paragraph.strip() == "":
                wrapped.append("")
            else:
                wrapped.extend(textwrap.wrap(paragraph, width=inner_width))

        # Footer: sender + timestamp
        timestamp = time.strftime("%H:%M")
        footer_text = f"{sender}  {timestamp}"

        # Build the bubble parts
        top = "\u250c" + "\u2500" * (bubble_width - 2) + "\u2510"       # top border
        bottom = "\u2514" + "\u2500" * (bubble_width - 2) + "\u2518"    # bottom border

        body_lines = []
        for line in wrapped:
            body_lines.append("\u2502 " + line.ljust(inner_width) + " \u2502")

        # Footer line (right-justified inside the bubble)
        body_lines.append("\u2502 " + footer_text.rjust(inner_width) + " \u2502")

        # Left-padding for right-aligned (sent) bubbles
        pad = " " * (term_width - bubble_width) if align == "right" else ""

        # Pick color
        color = ""
        dim = ""
        reset = ""
        if cls._use_color():
            color = cls._GREEN if align == "right" else cls._CYAN
            dim = cls._DIM
            reset = cls._RESET

        # Print the bubble
        print(f"{pad}{color}{top}{reset}")
        for i, line in enumerate(body_lines):
            if i == len(body_lines) - 1:
                # Footer line is dimmer
                print(f"{pad}{color}{dim}{line}{reset}")
            else:
                print(f"{pad}{color}{line}{reset}")
        print(f"{pad}{color}{bottom}{reset}")
        print()  # spacing between bubbles

    @classmethod
    def print_status(cls, message: str):
        """Print a centered WhatsApp-style status/system message."""
        term_width = min(shutil.get_terminal_size((120, 24)).columns, 120)
        dim = cls._DIM if cls._use_color() else ""
        reset = cls._RESET if cls._use_color() else ""
        centered = message.center(term_width)
        print(f"{dim}{centered}{reset}")

#!/usr/bin/env python3
"""
Simple Test Client for Assistant Server

This client allows you to interact with the assistant.py HTTP server.

USAGE:
    # Single message
    python test_client.py "Please read the file seed.txt"
    
    # Interactive mode
    python test_client.py --interactive
    
    # Custom server URL
    python test_client.py --url http://localhost:8000 "List all files"
"""

import sys
import argparse
import requests
from typing import Optional


class AssistantClient:
    """Simple HTTP client for the assistant server."""
    
    def __init__(self, url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            url: Base URL of the assistant server
        """
        self.url = url.rstrip('/')
        self.chat_endpoint = f"{self.url}/chat"
        self.health_endpoint = f"{self.url}/health"
    
    def check_health(self) -> bool:
        """
        Check if the server is healthy.
        
        Returns:
            True if server is responding, False otherwise
        """
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                print(f"✓ Server is healthy: {response.json()}")
                return True
            else:
                print(f"✗ Server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to server at {self.url}")
            print("  Make sure the server is running: python assistant.py --server")
            return False
        except Exception as e:
            print(f"✗ Error checking health: {e}")
            return False
    
    def send_message(self, message: str) -> Optional[str]:
        """
        Send a message to the assistant.
        
        Args:
            message: The message to send
            
        Returns:
            The assistant's response, or None if error
        """
        try:
            print(f"\n{'='*60}")
            print(f"YOU: {message}")
            print(f"{'='*60}")
            
            response = requests.post(
                self.chat_endpoint,
                json={"message": message},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result.get("response", "")
                
                print(f"\nASSISTANT:")
                print(assistant_response)
                print(f"{'='*60}\n")
                
                return assistant_response
            else:
                print(f"\n✗ Error: Server returned status {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            print(f"\n✗ Cannot connect to server at {self.url}")
            print("  Make sure the server is running: python assistant.py --server")
            return None
        except requests.exceptions.Timeout:
            print(f"\n✗ Request timed out")
            return None
        except Exception as e:
            print(f"\n✗ Error: {e}")
            return None
    
    def interactive_mode(self):
        """Run in interactive mode - continuous conversation."""
        print("=" * 60)
        print("  Interactive Assistant Client")
        print("=" * 60)
        print(f"Connected to: {self.url}")
        print("Type your messages. Commands:")
        print("  'exit' or 'quit' - Exit the client")
        print("  'health' - Check server health")
        print("=" * 60)
        print()
        
        # Check health first
        if not self.check_health():
            print("\nWarning: Server is not responding. Starting anyway...")
        
        print()
        
        while True:
            try:
                # Get user input
                message = input("You: ").strip()
                
                # Check for empty input
                if not message:
                    continue
                
                # Check for exit commands
                if message.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                # Check for health command
                if message.lower() == 'health':
                    self.check_health()
                    continue
                
                # Send message
                self.send_message(message)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            
            except EOFError:
                print("\n\nEnd of input. Goodbye!")
                break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simple client for assistant.py HTTP server"
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="Message to send to the assistant (if not in interactive mode)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Assistant server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = AssistantClient(args.url)
    
    # Interactive mode
    if args.interactive:
        client.interactive_mode()
        return
    
    # Single message mode
    if args.message:
        # Check health first
        if not client.check_health():
            sys.exit(1)
        
        # Send message
        response = client.send_message(args.message)
        
        if response is None:
            sys.exit(1)
        
        sys.exit(0)
    
    # No message provided - show usage
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()

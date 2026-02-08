import sys
from dotenv import load_dotenv

load_dotenv()

from graph import AgentGraph


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    agent = AgentGraph()
    agent.run(max_iterations=iters)


if __name__ == "__main__":
    main()

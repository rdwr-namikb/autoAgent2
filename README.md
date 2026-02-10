# AutoAgent - Behavioral Learning Agent with LangGraph

An autonomous learning agent built on **LangGraph** that iteratively crafts prompts, sends them to a target assistant server, decodes obfuscated responses, and tracks progress using RAG-based long-term memory.

## How It Works

The agent runs a looping state graph that repeats until it succeeds or hits the iteration limit:

```
START --> call_assistant --> analyze_response --> log_iteration
              ^                                       |
              |                                 [continue?]
              |                                   /     \
         generate_payload <-- yes             no --> END
```

Each iteration:

1. **call_assistant** -- sends the current payload to the target HTTP server.
2. **analyze_response** -- checks the raw response for encoding signals; if detected, runs the decode pipeline (hex, base64, Caesar cipher, Morse code, URL encoding, LLM fallback) before analysis. Then measures progress via cosine similarity against a ground truth, updates the strategy, and saves the experience to RAG memory.
3. **log_iteration** -- writes the iteration details to a JSON log file.
4. **should_continue** -- decides whether to loop (generate a new payload) or stop.
5. **generate_payload** -- uses RAG memory and the current strategy to craft a new prompt via the LLM.

## Project Structure

```
autoAgent2/
├── main.py            # Entry point -- parses CLI args, runs the agent
├── graph.py           # AgentGraph class -- builds and executes the LangGraph workflow
├── nodes.py           # AgentNodes class -- all graph node functions
├── state.py           # AgentState TypedDict -- shared state schema
├── llm_handler.py     # LLMHandler class -- LLM calls, embeddings, cosine similarity
├── decoders.py        # ResponseDecoder class -- hex/base64/Caesar/Morse/URL/LLM decoders
├── display.py         # Display abstraction -- CLIDisplay, WebDisplay, DisplayDispatcher
├── web_server.py      # FastAPI server for the WhatsApp-style Web UI
├── static/
│   └── index.html     # Single-page WhatsApp-style chat frontend
├── custom_rag.py      # ExperienceMemory class -- FAISS-backed RAG memory bank
├── config.py          # Loads environment variables from .env
├── requirements.txt   # Python dependencies
├── .env.example       # Template for required environment variables
└── .gitignore
```

## Prerequisites

- Python 3.10+
- A target assistant server running at `http://localhost:8000` (e.g. `python assistant.py --server`)
- API keys for your LLM provider and OpenAI (for embeddings)

## Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd autoAgent2
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and fill in your keys:

   | Variable | Description |
   |----------|-------------|
   | `LLM_API_KEY` | API key for the LLM used by the learning agent |
   | `LLM_MODEL` | Model name (e.g. `gpt-4o`, `deepseek-chat`) |
   | `LLM_BASE_URL` | Base URL of the LLM API (optional, defaults to DeepSeek) |
   | `OPENAI_API_KEY` | OpenAI key for `text-embedding-3-large` embeddings |
   | `GROUND_TRUTH_KEY` | The expected target key used to measure progress |

## Usage

Start the target assistant server first, then run the agent:

### CLI Mode (default)

```bash
# Default: up to 1000 iterations
python3 main.py

# Custom iteration limit
python3 main.py 500
```

The agent prints a WhatsApp-style chat log in the terminal and writes detailed iteration history to `learning_history_langgraph.json`.

### Web UI Mode

```bash
# Launch the web UI (opens browser automatically)
python3 main.py --mode web

# Custom port
python3 main.py --mode web --port 9090
```

Opens a WhatsApp-style chat interface in your browser at `http://localhost:8080`. Use the **Start Agent** / **Stop Agent** buttons to control the run. Messages stream in real time via WebSocket.

## Architecture

| Class | File | Responsibility |
|-------|------|----------------|
| `AgentState` | `state.py` | TypedDict defining the shared state passed between all nodes |
| `DisplayDispatcher` | `display.py` | Abstract base for display backends; `CLIDisplay` for terminal, `WebDisplay` for browser |
| `LLMHandler` | `llm_handler.py` | Manages LLM invocation, embedding generation, cosine similarity, and the RAG memory bank |
| `ResponseDecoder` | `decoders.py` | Programmatic decoders (hex, decimal, base64, URL, Caesar, Morse) plus an LLM-based fallback |
| `AgentNodes` | `nodes.py` | All LangGraph node functions; receives `LLMHandler` and `ResponseDecoder` via constructor |
| `AgentGraph` | `graph.py` | Wires everything together into a compiled `StateGraph` and orchestrates execution |
| `ExperienceMemory` | `custom_rag.py` | FAISS vector store for saving and retrieving past experiences |
| Web Server | `web_server.py` | FastAPI app with WebSocket endpoint, agent start/stop API, static file serving |

## License

This project is provided as-is for research and educational purposes.

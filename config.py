import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration (DeepSeek)
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")

# OpenAI key for embeddings (DeepSeek doesn't support embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ground Truth Key (optional)
GROUND_TRUTH_KEY = os.getenv("GROUND_TRUTH_KEY", "")

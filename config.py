import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# Ground Truth Key (optional)
GROUND_TRUTH_KEY = os.getenv("GROUND_TRUTH_KEY", "")

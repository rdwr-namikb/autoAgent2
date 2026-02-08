from typing import List, Dict, TypedDict


class AgentState(TypedDict):
    iteration: int
    max_iterations: int
    payload: str
    response: str
    decoded_response: str  # Response after decoding any obfuscation (hex, base64, etc.)
    strategy: str
    progress: float
    last_progress: float
    best_progress: float
    history: List[Dict]
    sent_payloads: List[str]
    success: bool
    lessons_learned: List[str]
    target_url: str
    log_file: str

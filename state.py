from typing import List, Dict, TypedDict, Optional


class AgentState(TypedDict):
    iteration: int
    max_iterations: int
    payload: str
    response: str
    decoded_response: str  # Response after decoding any obfuscation (hex, base64, etc.)
    helper_decoded: bool  # True if agent-helper performed decoding
    strategy: str
    suggested_encoding: Optional[str]  # Encoding strategy suggested by analyzer
    progress: float
    last_progress: float
    best_progress: float
    history: List[Dict]
    sent_payloads: List[str]
    success: bool
    lessons_learned: List[str]
    stall_count: int  # consecutive iterations with no progress improvement
    target_url: str
    log_file: str

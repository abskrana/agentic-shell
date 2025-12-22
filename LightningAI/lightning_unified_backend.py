"""
Lightning AI Unified Backend
A single FastAPI service that provides:
1. Qwen/Gemini AI model inference
2. Google Speech-to-Text
3. Log storage
4. Centralized configuration
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import os
from datetime import datetime
from pathlib import Path
import base64

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"Could not load .env: {e}")

# Initialize FastAPI
app = FastAPI(
    title="Agentic Shell Unified Backend",
    version="1.0.0",
    description="Unified backend for AI inference, speech-to-text, and logging"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Configuration
# ========================================
# Logging Configuration
LOGS_COLLECTION_PATH = Path(os.getenv('LOGS_COLLECTION_PATH', 'Logs/logs_collections.jsonl'))
CURRENT_LOGS_PATH = Path(os.getenv('CURRENT_LOGS_PATH', 'Logs/logs_1.jsonl'))

# Create Logs directory if it doesn't exist
LOGS_COLLECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
CURRENT_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)

# AI Model Configuration
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY', '')
QWEN_MODEL_NAME = os.getenv('QWEN_MODEL_NAME', '')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', '')

# Google Cloud Configuration for Speech-to-Text
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')

# Print configuration for debugging
print("Lightning AI Unified Backend Configuration")
print(f"GEMINI_API_KEY: {'✓ Set' if GEMINI_API_KEY else '✗ Not set'}")
print(f"GEMINI_MODEL_NAME: {GEMINI_MODEL_NAME}")
print(f"QWEN_MODEL_NAME: {QWEN_MODEL_NAME}")
print(f"GOOGLE_APPLICATION_CREDENTIALS: {'✓ Set' if GOOGLE_APPLICATION_CREDENTIALS else '✗ Not set'}")
print(f"LOGS_COLLECTION_PATH: {LOGS_COLLECTION_PATH}")
print(f"CURRENT_LOGS_PATH: {CURRENT_LOGS_PATH}")
print("="*60 + "\n")

# Initialize AI Models
gemini_model = None
qwen_model = None
speech_client = None

def initialize_gemini_model():
    """Initialize the Gemini model"""
    global gemini_model
    
    print("🤖 Initializing Gemini model...")
    if not GEMINI_API_KEY:
        print("✗ GOOGLE_API_KEY not set")
        gemini_model = None
        return
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        print(f"✓ Gemini model loaded: {GEMINI_MODEL_NAME}")
    except Exception as e:
        print(f"✗ Failed to load Gemini model: {e}")
        gemini_model = None

def initialize_qwen_model():
    """Initialize the Qwen model"""
    global qwen_model
    
    print("🤖 Initializing Qwen model...")
    try:
        from qwen_adapter import GenerativeModel
        qwen_model = GenerativeModel(model_name=QWEN_MODEL_NAME)
        print(f"✓ Qwen model loaded: {QWEN_MODEL_NAME}")
    except Exception as e:
        print(f"✗ Failed to load Qwen model: {e}")
        qwen_model = None

def initialize_speech_client():
    """Initialize Google Speech-to-Text client"""
    global speech_client
    
    if not GOOGLE_APPLICATION_CREDENTIALS:
        print("Speech-to-Text: GOOGLE_APPLICATION_CREDENTIALS not set")
        return
    
    try:
        from google.cloud import speech
        speech_client = speech.SpeechClient()
        print("✓ Speech-to-Text client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize Speech-to-Text: {e}")
        speech_client = None

@app.on_event("startup")
async def startup_event():
    initialize_gemini_model()
    initialize_qwen_model()
    initialize_speech_client()

# ========================================
# Logging Helper Functions
# ========================================

def write_to_collection_log(entry: dict):
    """Write log entry to the collection log file"""
    try:
        with open(LOGS_COLLECTION_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"Error writing to collection log: {e}")
        return False

def write_to_current_log(entry: dict):
    """Write log entry to the current log file"""
    try:
        with open(CURRENT_LOGS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"Error writing to current log: {e}")
        return False

def count_logs_in_file(file_path: Path) -> int:
    """Count the number of log entries in a file"""
    if not file_path.exists():
        return 0
    try:
        with open(file_path, "r") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0

# ========================================
# Data Models
# ========================================

class LogEntry(BaseModel):
    """Log entry schema"""
    timestamp: str
    session_id: str
    prompt: str
    mode: str
    status: str
    model: Optional[str] = None
    language: Optional[str] = None
    plan: Optional[List[Dict[str, Any]]] = None
    executed_commands: Optional[List[Dict[str, Any]]] = None
    answer: Optional[str] = None
    error: Optional[str] = None

class AIPromptRequest(BaseModel):
    """AI model prompt request"""
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    model: Optional[str] = None  # 'gemini' or 'qwen', if None uses default

class SpeechToTextRequest(BaseModel):
    """Speech-to-text request"""
    audio_content: str  # Base64 encoded audio
    language_code: str = "en-US"
    sample_rate: Optional[int] = 48000
    context_phrases: Optional[List[str]] = None
    context_boost: Optional[float] = 15.0

# ========================================
# AI Inference Endpoint
# ========================================

@app.post("/api/ai/generate")
async def generate_ai_response(request: AIPromptRequest):
    """
    Generate AI response using configured model (Qwen or Gemini)
    """
    # Determine which model to use
    model_choice = request.model or "gemini"  # Default to gemini
    
    # Select the appropriate model
    if model_choice.lower() == "qwen":
        selected_model = qwen_model
        model_name = QWEN_MODEL_NAME
        backend = "qwen"
    else:
        selected_model = gemini_model
        model_name = GEMINI_MODEL_NAME
        backend = "gemini"
    
    if selected_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"{backend.capitalize()} model not initialized. Check backend configuration."
        )
    
    try:
        # Generate response
        response = selected_model.generate_content(
            request.prompt,
            generation_config={
                "temperature": request.temperature,
                "max_output_tokens": request.max_tokens
            }
        )
        text = response.text
        
        return {
            "status": "success",
            "text": text,
            "model": model_name,
            "backend": backend
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI generation failed: {str(e)}"
        )

# ========================================
# Speech-to-Text Endpoint
# ========================================

@app.post("/api/speech/transcribe")
async def transcribe_speech(request: SpeechToTextRequest):
    """
    Transcribe audio to text using Google Speech-to-Text with enhanced configuration
    """
    if speech_client is None:
        raise HTTPException(
            status_code=503,
            detail="Speech-to-Text not initialized. Check GOOGLE_APPLICATION_CREDENTIALS."
        )
    
    try:
        # Decode base64 audio
        audio_content = base64.b64decode(request.audio_content)
        
        # Configure recognition with enhanced settings
        from google.cloud import speech
        audio = speech.RecognitionAudio(content=audio_content)
        
        # Build speech contexts if provided
        speech_contexts = []
        if request.context_phrases:
            speech_contexts.append(
                speech.SpeechContext(
                    phrases=request.context_phrases,
                    boost=request.context_boost
                )
            )
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=request.sample_rate,
            language_code=request.language_code,
            enable_automatic_punctuation=True,
            model='command_and_search',
            enable_word_confidence=True,
            use_enhanced=True,
            profanity_filter=False,
            enable_spoken_punctuation=True,
            enable_spoken_emojis=False,
            speech_contexts=speech_contexts if speech_contexts else None,
        )
        
        # Perform transcription
        response = speech_client.recognize(config=config, audio=audio)
        
        # Extract transcript
        transcript = ""
        confidence = 0.0
        
        if response.results:
            for result in response.results:
                if result.alternatives:
                    transcript += result.alternatives[0].transcript
                    confidence = max(confidence, result.alternatives[0].confidence)
        
        return {
            "status": "success",
            "transcript": transcript,
            "confidence": confidence,
            "language": request.language_code
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

# ========================================
# Logging Endpoints
# ========================================

@app.post("/api/logs")
async def receive_log(entry: LogEntry):
    """Receive and store a log entry in both collection and current log files"""
    try:
        entry_dict = entry.model_dump()
        
        # Write to collection log
        collection_success = write_to_collection_log(entry_dict)
        print(count_logs_in_file(LOGS_COLLECTION_PATH))
        print(entry_dict)
        
        # Write to current log
        current_success = write_to_current_log(entry_dict)
        print(count_logs_in_file(CURRENT_LOGS_PATH))
        
        if not collection_success or not current_success:
            raise Exception("Failed to write to one or more log files")

        # call the updater with current live incoming dict
        run_dashboard_update(entry_dict)

        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": entry.session_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store log: {str(e)}"
        )

import json, io
from pathlib import Path

# -------------------- GLOBALS --------------------
_STATE_PATH = Path("dashboard_state.json")
_TRACK = ["model", "mode", "status", "language"]
_COLORS = ["#3B82F6", "#EF4444", "#10B981", "#A855F7"]
_STATE = {}
_LAST = {}
_RUN = None
_CALLS = 0


# ==========================================================
# 1️⃣ INITIALIZATION
# ==========================================================
def init_dashboard_wandb(project="logs-dashboard"):
    """Initialize dashboard state and W&B run once."""
    global _STATE, _LAST, _RUN

    try:
        import wandb
    except ImportError:
        raise RuntimeError("Please install wandb: pip install wandb")

    # Load or create state file
    if _STATE_PATH.exists():
        try:
            _STATE = json.loads(_STATE_PATH.read_text(encoding="utf-8") or "{}")
        except Exception:
            _STATE = {}
    for k in _TRACK:
        _STATE.setdefault(k, {})

    # Copy snapshot for diff
    _LAST = {k: dict(v) for k, v in _STATE.items()}

    # Init W&B once
    _RUN = wandb.init(project=project, reinit=True)
    print(f"✅ Initialized W&B dashboard: {project}")
    return True


# ==========================================================
# 2️⃣ UPDATE / RUN LOOP
# ==========================================================
def run_dashboard_update(entry_dict):
    """
    Called every second.
    Updates counts and logs only changed bar plots to W&B.
    """
    global _STATE, _LAST, _RUN, _CALLS
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import wandb

    def norm(v):
        if v is None:
            return "unknown"
        if isinstance(v, str):
            s = v.strip()
            return s if s else "unknown"
        return str(v)

    changed = []

    # --- Update counts ---
    for k in _TRACK:
        val = norm(entry_dict.get(k))
        _STATE[k][val] = _STATE[k].get(val, 0) + 1
        if _STATE[k] != _LAST.get(k, {}):
            changed.append(k)

    # --- Save persistent state every few updates ---
    _CALLS += 1
    if _CALLS % 10 == 0:
        try:
            _STATE_PATH.write_text(json.dumps(_STATE, indent=2), encoding="utf-8")
        except Exception:
            pass

    # --- Skip if nothing changed ---
    if not changed:
        return {"ok": True, "changed": []}

    payload = {}

    # --- Create compact bar plots ---
    for i, key in enumerate(changed):
        items = sorted(_STATE[key].items(), key=lambda x: x[1], reverse=True)
        labels = [str(lbl) for lbl, _ in items]
        values = [v for _, v in items]
        color = _COLORS[i % len(_COLORS)]

        fig, ax = plt.subplots(figsize=(6.5, 2.2))
        if len(labels) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray")
            ax.set_xticks([]); ax.set_yticks([])
        elif len(labels) > 8:
            y = np.arange(len(labels))
            bars = ax.barh(y, values, color=color, edgecolor="black", linewidth=0.3)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            for b, v in zip(bars, values):
                ax.text(b.get_width() + 0.3, b.get_y() + b.get_height()/2, str(v),
                        ha="left", va="center", fontsize=8)
        else:
            x = np.arange(len(labels))
            bars = ax.bar(x, values, color=color, edgecolor="black", linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, str(v),
                        ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{key.capitalize()} Distribution", fontsize=10)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
        plt.tight_layout(pad=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=160, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        buf.seek(0)

        payload[f"{key}_bar"] = wandb.Image(Image.open(buf).convert("RGB"),
                                            caption=f"{key.capitalize()} Distribution")
        _LAST[key] = dict(_STATE[key])

    # --- Log to W&B ---
    try:
        wandb.log(payload, commit=True)
    except Exception as e:
        print("⚠️ Failed to log to W&B:", e)

    return {"ok": True, "changed": changed}


# ========================================
# Run Server
# ========================================

if __name__ == "__main__":
    init_dashboard_wandb("logs-dashboard")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

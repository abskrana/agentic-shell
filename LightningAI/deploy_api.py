# deploy_api.py (Final Version for Merged Model)

import subprocess
import os

# --- Configuration ---
# Point to the new merged model directory
# MERGED_MODEL_DIR = "./models/latest-merged" # Updated for 7B model output

# HOST = "0.0.0.0"
# PORT = 8088

def deploy():
    MERGED_MODEL_DIR = "./models/latest-merged" # Updated for 7B model output

    HOST = "0.0.0.0"
    PORT = 8088

    if not os.path.exists(MERGED_MODEL_DIR):
        print(f"❌ Error: Merged model directory not found at '{MERGED_MODEL_DIR}'")
        print("Please make sure you have run the merge_and_save.py script successfully.")
        return

    print(f"🚀 Starting vLLM API server for merged model at {MERGED_MODEL_DIR}...")
    
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MERGED_MODEL_DIR,
        "--host", HOST,
        "--port", str(PORT),
        "--dtype", "auto",  # Auto-detect best precision (bfloat16 for 7B)
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "12288",
        "--trust-remote-code",
        "--enable-prefix-caching",  # Cache common prompts (HUGE speedup)
        "--disable-log-requests",  # Reduce overhead
        "--max-num-seqs", "16",  # Handle more concurrent requests
        "--tensor-parallel-size", "1",  # Single GPU optimization
    ]

    print(f"Running command: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"🔥 vLLM server failed to start: {e}")
    except KeyboardInterrupt:
        print("🛑 Server stopped by user.")

if __name__ == "__main__":
    deploy()
import os
import time
import gc
import torch
import traceback

# your imports
from retraining import retrain_main
from merge_and_save import merge_main
from deploy_api import deploy

# optional: lazy import wandb to avoid import errors at module load
try:
    import wandb
except Exception:
    wandb = None


# -------------------------
# Helper: ensure a wandb run
# -------------------------
def ensure_wandb_run(project="model-pipeline", entity=None):
    """
    Ensure wandb.run exists. If wandb is not available, do nothing.
    If there's no active run, create one (reinit=True).
    Return the active run or None.
    """
    global wandb
    if wandb is None:
        try:
            import wandb as _wandb
            wandb = _wandb
        except Exception:
            # wandb not installed / import failed
            return None

    # If a run already exists, return it
    if getattr(wandb, "run", None) is not None:
        return wandb.run

    # Otherwise try to init a new run (best-effort)
    try:
        run = wandb.init(project=project, entity=entity, reinit=True)
        return run
    except Exception as e:
        # initialization failed; print and continue without W&B
        print("⚠️ ensure_wandb_run: wandb.init() failed:", e)
        return None


# -------------------------
# Initialization
# -------------------------
def init_pipeline(project="model-pipeline", entity=None):
    """Init environment info and a wandb run (uses ensure_wandb_run)."""
    print("🔧 Initializing pipeline...")

    run = ensure_wandb_run(project=project, entity=entity)

    # If wandb is available and run present, log a start alert & system info
    if run is not None:
        try:
            wandb.alert(title="🚀 Pipeline Start", text="Pipeline execution started.", level=wandb.AlertLevel.INFO)
        except Exception as e:
            print("⚠️ wandb.alert failed at init:", e)

        try:
            gpu_available = torch.cuda.is_available()
            device = torch.cuda.get_device_name(0) if gpu_available else "CPU"
            mem = torch.cuda.memory_allocated(0) / 1e9 if gpu_available else 0.0

            wandb.log({
                "system/device": device,
                "system/gpu_available": gpu_available,
                "system/gpu_mem_gb": mem,
                "status": "initialized"
            })
        except Exception as e:
            print("⚠️ wandb.log failed at init:", e)

        print(f"✅ Initialized W&B run: {run.name}")
        print(f"💻 Device: {device} | GPU Mem: {mem:.2f} GB")
    else:
        print("⚠️ W&B not available — continuing without wandb.")

    return run


# -------------------------
# Main runner
# -------------------------
def run_pipeline():
    run = None
    try:
        run = init_pipeline(project="model-pipeline")

        print("\n==============================")
        print("🚀 STARTING FULL PIPELINE RUN")
        print("==============================\n")

        # ensure run before alert/log
        ensure_wandb_run()
        try:
            if wandb is not None and wandb.run is not None:
                wandb.alert(title="🧠 Retraining", text="Retraining started.", level=wandb.AlertLevel.INFO)
        except Exception as e:
            print("⚠️ wandb.alert failed:", e)

        # ---- retrain ----
        retrain_main()
        print("✅ Retraining completed.")

        # make sure a run still exists before logging
        ensure_wandb_run()
        try:
            if wandb is not None and wandb.run is not None:
                wandb.log({"stage": "retrain_complete"})
        except Exception as e:
            print("⚠️ wandb.log after retrain failed (continuing):", e)

        # ---- merge ----
        merge_main()
        print("✅ Merge and save completed.")

        ensure_wandb_run()
        try:
            if wandb is not None and wandb.run is not None:
                wandb.log({"stage": "merge_complete"})
        except Exception as e:
            print("⚠️ wandb.log after merge failed (continuing):", e)

        # GPU cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)
        print("🧹 GPU memory cleared.")

        ensure_wandb_run()
        try:
            if wandb is not None and wandb.run is not None:
                wandb.log({"gpu_cleared": True})
                wandb.alert(title="🧹 Memory Cleared", text="GPU memory successfully cleared before deployment.", level=wandb.AlertLevel.INFO)
        except Exception as e:
            print("⚠️ wandb logging/alert for memory clear failed:", e)

        # ---- deploy ----
        ensure_wandb_run()
        try:
            if wandb is not None and wandb.run is not None:
                wandb.log({"stage": "deploy_complete", "status": "success"})
                wandb.alert(title="✅ Pipeline Completed", text="Full retrain → merge → deploy pipeline completed successfully!", level=wandb.AlertLevel.SUCCESS)
        except Exception as e:
            print("⚠️ final wandb.log/alert failed:", e)

        print("🎉 All tasks finished successfully!")
        deploy()
        print("✅ Deployment completed.")


    except Exception as e:
        print("❌ Pipeline failed:", str(e))
        traceback.print_exc()

        # Ensure we have a run before sending failure alert/log
        ensure_wandb_run()
        try:
            if wandb is not None and wandb.run is not None:
                wandb.alert(
                    title="❌ Pipeline Failed",
                    text=f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}",
                    level=wandb.AlertLevel.ERROR,
                )
                wandb.log({"status": "failed", "error": str(e)})
        except Exception as log_error:
            print("⚠️ W&B failure alert/log failed:", log_error)

    finally:
        # final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # finish/close run if present
        try:
            if wandb is not None and wandb.run is not None:
                wandb.finish()
        except Exception as e:
            print("⚠️ wandb.finish() failed:", e)

        print("🧩 Pipeline finished (success or fail). GPU memory cleared.")


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    run_pipeline()

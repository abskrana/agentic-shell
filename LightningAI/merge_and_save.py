import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def merge_and_save(base_model: str, adapter_dir: str, merged_model_dir: str):
    """Merge a LoRA adapter into a base model and save the merged model."""
    if not os.path.exists(adapter_dir):
        print(f"❌ Error: Adapter directory not found at '{adapter_dir}'")
        exit(1)

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)

    print("Merging adapter into the base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {merged_model_dir}")
    model.save_pretrained(merged_model_dir, safe_serialization=False)

    tokenizer.save_pretrained(merged_model_dir)

    print("✅ Merging complete! Your standalone fine-tuned model is ready.")
    try:
        file_path = '/teamspace/studios/this_studio/Logs/logs_1.jsonl'
        with open(file_path, 'w') as f:
                pass
        print(f"Successfully cleared data from: {file_path}")
    except IOError as e:
        print(f"Error accessing file {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



def merge_main():
    # --- Configuration ---
    BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Updated for 7B model
    ADAPTER_DIR = "./models/latest"                 # LoRA fine-tuned output
    MERGED_MODEL_DIR = "./models/latest-merged"     # Output of merged model

    merge_and_save(BASE_MODEL, ADAPTER_DIR, MERGED_MODEL_DIR)


if __name__ == "__main__":
    merge_main()

# train_bash_agent.py (Updated for jiacheng-ye/nl2bash dataset)

import os
import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Configuration for 7B Model with new dataset ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
# --- CHANGE 1: Update the dataset name ---
DATASET_NAME = "jiacheng-ye/nl2bash"
OUTPUT_DIR = "./models/bash-agent-qwen2.5-coder-7B-v1" # Updated for 7B model
WANDB_PROJECT = "bash-agent-finetuning"
WANDB_RUN_NAME = "qwen2.5-coder-7b-nl2bash" # Updated run name

class BashAgentTrainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "model": MODEL_NAME, "dataset": DATASET_NAME, "batch_size": 4,
                "learning_rate": 2e-4, "epochs": 3, "lora_r": 16, "lora_alpha": 32,
            }
        )

    def create_chat_prompt(self, instruction: str, output: str = None) -> str:
        messages = [
            {"role": "system", "content": "You are an expert AI assistant. Convert the user's natural language instruction into a precise bash command."},
            {"role": "user", "content": instruction},
        ]
        if output:
            messages.append({"role": "assistant", "content": output})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True if not output else False)

    # --- CHANGE 2: Major simplification of data loading ---
    def load_and_prepare_dataset(self):
        print(f"📥 Loading {DATASET_NAME} dataset...")
        
        # Load the pre-defined train and validation splits directly
        train_data = load_dataset(DATASET_NAME, split="train")
        validation_data = load_dataset(DATASET_NAME, split="validation")

        train_data = train_data.select(range(min(1000, len(train_data))))
        print(f"✂️ Using only the first {len(train_data)} training examples for fine-tuning.")

        # Function to apply the chat template to each example
        def format_dataset(example):
            return {"text": self.create_chat_prompt(example['nl'], example['bash'])}

        # Apply the formatting
        self.train_dataset = train_data.map(format_dataset)
        self.eval_dataset = validation_data.map(format_dataset)
        
        print(f"✅ Dataset prepared: {len(self.train_dataset)} training, {len(self.eval_dataset)} validation examples.")
        wandb.log({"dataset_size": len(self.train_dataset), "eval_size": len(self.eval_dataset)})

    def load_model_and_tokenizer(self):
        print(f"📦 Loading model: {MODEL_NAME}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map="auto", 
            torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded: {trainable_params:,} / {total_params:,} trainable parameters ({100 * trainable_params / total_params:.2f}%)")
        wandb.log({"trainable_params": trainable_params, "total_params": total_params})

    def train(self):
        print("🚀 Starting training...")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR, num_train_epochs=3, per_device_train_batch_size=4,
            gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type="cosine",
            warmup_ratio=0.03, logging_steps=10, evaluation_strategy="steps", eval_steps=100,
            save_strategy="steps", save_steps=500, save_total_limit=2, 
            bf16=True, optim="paged_adamw_8bit", report_to="wandb", run_name=WANDB_RUN_NAME,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
        )
        trainer = SFTTrainer(
            model=self.model, args=training_args, train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset, tokenizer=self.tokenizer,
            dataset_text_field="text", max_seq_length=1024, packing=False,
        )
        trainer.train()
        print(f"💾 Saving final LoRA adapter to {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR)
        self.tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ Training complete! Model saved to {OUTPUT_DIR}")
        wandb.finish()

def main():
    trainer = BashAgentTrainer()
    print("Step 1: Loading base model and tokenizer...")
    trainer.load_model_and_tokenizer() 
    print("\nStep 2: Loading and preparing dataset...")
    trainer.load_and_prepare_dataset()
    print("\nStep 3: Starting fine-tuning...")
    trainer.train()
    print("🎉 All done! Your fine-tuned model is ready to deploy.")

if __name__ == "__main__":
    main()
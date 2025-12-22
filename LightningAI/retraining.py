import os
import shutil
import torch
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
import json

MODEL_NAME = "./models/latest"
OLD_MODEL = "./models/old"
OUTPUT_DIR = "./models/latest"
WANDB_PROJECT = "bash-agent-finetuning"
WANDB_RUN_NAME = "bash-agent-retrain"

class BashAgentTrainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={"model": MODEL_NAME, "batch_size": 4, "learning_rate": 2e-4, "epochs": 3, "lora_r": 16, "lora_alpha": 32})

    def create_chat_prompt(self, instruction: str, output: str = None) -> str:
        messages = [
            {"role": "system", "content": "You are an expert AI assistant. Convert the user's natural language instruction into a precise bash command."},
            {"role": "user", "content": instruction}
        ]
        if output:
            messages.append({"role": "assistant", "content": output})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True if not output else False)

    def load_and_prepare_dataset(self):
        df = dataset()
        train_split = 0.9
        split_idx = int(len(df) * train_split)
        train_df = df.iloc[:split_idx]
        eval_df = df.iloc[split_idx:]
        def format_row(example):
            return {"text": self.create_chat_prompt(example["nl"], example["bash"])}
        self.train_dataset = Dataset.from_pandas(train_df).map(format_row)
        self.eval_dataset = Dataset.from_pandas(eval_df).map(format_row)
        wandb.log({"train_size": len(self.train_dataset), "eval_size": len(self.eval_dataset)})

    def load_model_and_tokenizer(self):
        base_fallback = "Qwen/Qwen2.5-Coder-7B-Instruct"

        if os.path.exists(os.path.join(MODEL_NAME, "config.json")):
            model_path = MODEL_NAME
            print("📦 Loading model from ./models/latest")
        elif os.path.exists(os.path.join(OLD_MODEL, "config.json")):
            model_path = OLD_MODEL
            print("📦 Loading model from ./models/old (fallback)")
        else:
            print(f"⚠️ No local model found. Downloading base model: {base_fallback}")
            model_path = base_fallback

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        print("✅ Model ready for training.")
        wandb.log({"model_source": model_path})

    def train(self):
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            bf16=True,
            optim="paged_adamw_8bit",
            report_to="wandb",
            run_name=WANDB_RUN_NAME,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=1024,
            packing=False
        )
        trainer.train()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.model.save_pretrained(OUTPUT_DIR)
        self.tokenizer.save_pretrained(OUTPUT_DIR)
        print("📦 New model saved in ./models/latest")
        wandb.finish()

def dataset():
    input_path = "Logs/logs_1.jsonl"
    output_path = "retraindata.csv"
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") != "completed":
                continue
            prompt = entry.get("prompt", "").strip()
            executed = entry.get("executed_commands", None)
            if not executed:
                continue
            commands = [cmd["command"].strip().replace("\n", " ") for cmd in executed if "command" in cmd]
            if not commands:
                continue
            bash_script = " && ".join(commands)
            rows.append({"nl": prompt, "bash": bash_script})
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, mode="w")
    return df

def handle_model_versioning():
    latest_path = "./models/latest"
    old_path = "./models/old"
    if os.path.exists(old_path):
        shutil.rmtree(old_path)
        print("🗑️ Removed previous 'old' model.")
    if os.path.exists(latest_path) and os.listdir(latest_path):
        shutil.move(latest_path, old_path)
        print("🔁 Moved 'latest' → 'old'.")
    else:
        print("ℹ️ No existing 'latest' model found; starting fresh.")
    os.makedirs(latest_path, exist_ok=True)

def retrain_main():
    handle_model_versioning()
    trainer = BashAgentTrainer()
    trainer.load_model_and_tokenizer()
    trainer.load_and_prepare_dataset()
    trainer.train()
    print("🎉 Retraining complete! latest = new model | old = previous")

if __name__ == "__main__":
    retrain_main()

    wandb.finish()


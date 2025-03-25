import os
from typing import Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from fs_gpt.data.JSONLStreamingDataset import JSONLStreamingDataset


class Tuner:
    def __init__(self, args: Dict) -> None:
        self.args = args
        self.model_name_or_path = args.get("model_name_or_path")
        self.output_dir = args.get("output_dir", f"logs/{(os.path.basename(self.model_name_or_path))}")

    def train(self) -> None:
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        train_dataset = JSONLStreamingDataset(self.args["train_dataset_names"], tokenizer=tokenizer, args=self.args,)
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=self.output_dir,
            overwrite_output_dir=self.args.get("overwrite_output_dir", False),
            num_train_epochs=self.args.get("num_train_epochs", 2),
            per_device_train_batch_size=self.args.get("per_device_train_batch_size", 1),
            learning_rate=self.args.get("learning_rate", 1e-5),
            lr_scheduler_type=self.args.get("lr_scheduler_type", "cosine"),
            warmup_ratio=self.args.get("warmup_ratio", 0.1),
            bf16=self.args.get("bf16", True),
            ddp_timeout=self.args.get("ddp_timeout", 180000000),
            save_steps=self.args.get("save_steps", 500),
            logging_steps=self.args.get("logging_steps", 10),
            per_device_eval_batch_size=self.args.get("per_device_eval_batch_size", 1),
            evaluation_strategy=self.args.get("evaluation_strategy", "steps"),
            eval_steps=self.args.get("eval_steps", 500),
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

def main(args: Dict):
    Tuner(args).train()

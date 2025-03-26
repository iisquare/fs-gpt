import os
from typing import Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from fs_gpt.data.JSONLDataset import JSONLDataset
from fs_gpt.data.JSONLStreamingDataset import JSONLStreamingDataset


class Tuner:
    def __init__(self, args: Dict) -> None:
        self.args = args
        self.model_name_or_path = args.get("model_name_or_path")
        self.output_dir = args.get("output_dir", f"logs/{(os.path.basename(self.model_name_or_path))}")
        self.max_steps = args.get("max_steps", -1)

    def train(self) -> None:
        print(f"Load model from {self.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        print(f"Load train dataset with {self.args['train_dataset_names']}")
        if self.max_steps == -1:
            train_dataset = JSONLDataset(self.args["train_dataset_names"], tokenizer=tokenizer, args=self.args,)
        else:
            train_dataset = JSONLStreamingDataset(self.args["train_dataset_names"], tokenizer=tokenizer, args=self.args)
        print(f"Load evaluate dataset with {self.args['eval_dataset_names']}")
        eval_dataset = JSONLDataset(self.args["eval_dataset_names"], tokenizer=tokenizer, args=self.args,)
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=self.output_dir,
            do_train=self.args.get("do_train", True),
            do_eval=self.args.get("do_eval", True),
            overwrite_output_dir=self.args.get("overwrite_output_dir", False),
            num_train_epochs=self.args.get("num_train_epochs", 3.0),
            max_steps=self.max_steps,
            per_device_train_batch_size=self.args.get("per_device_train_batch_size", 1),
            learning_rate=self.args.get("learning_rate", 1e-5),
            lr_scheduler_type=self.args.get("lr_scheduler_type", "cosine"),
            warmup_ratio=self.args.get("warmup_ratio", 0.1),
            bf16=self.args.get("bf16", True),
            ddp_timeout=self.args.get("ddp_timeout", 1800),
            save_steps=self.args.get("save_steps", 500),
            logging_steps=self.args.get("logging_steps", 10),
            per_device_eval_batch_size=self.args.get("per_device_eval_batch_size", 1),
            eval_strategy=self.args.get("eval_strategy", "steps"),
            eval_steps=self.args.get("eval_steps", 500),
            deepspeed=self.args.get("deepspeed"),
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        print(f"Train...")
        trainer.train()
        print(f"Evaluate...")
        trainer.evaluate()
        print(f"Save model to {self.output_dir}")
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        print(f'Model saved at "{self.output_dir}"')

def main(args: Dict):
    Tuner(args).train()

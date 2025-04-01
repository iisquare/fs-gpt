import os
import math
from abc import abstractmethod
from typing import Dict, Optional, Union, List, Any

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, BitsAndBytesConfig

from fs_gpt.data.DatasetConfig import DatasetConfig
from fs_gpt.data.JSONLDataset import JSONLDataset
from fs_gpt.data.JSONLStreamingDataset import JSONLStreamingDataset
from fs_gpt.utils import ModelUtil, PlotUtil


class Tuner:
    def __init__(self, args: Dict) -> None:
        self.args = args
        self.model_name_or_path = args.get("model_name_or_path")
        self.output_dir = args.get("output_dir", f"logs/{(os.path.basename(self.model_name_or_path))}")
        self.max_steps = args.get("max_steps", -1)
        self.fine_tuning = args.get("fine_tuning")
        self.quantization_method = args.get("quantization_method")
        self.quantization_bit = args.get("quantization_bit", 4)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def train(self) -> None:
        print(f"Load model from {self.model_name_or_path}")
        model = self.model()
        model.config.use_cache = self.args.get("use_cache", True) # 可通过禁用KVCache节省显存
        print(f"Load train dataset with {self.args['train_dataset_names']}")
        if self.max_steps == -1:
            train_dataset = JSONLDataset(self.args["train_dataset_names"], tuner=self, args=self.args,)
        else: # 流式加载数据，必须指定max_steps最大步长
            train_dataset = JSONLStreamingDataset(self.args["train_dataset_names"], tuner=self, args=self.args)
        print(f"Load evaluate dataset with {self.args['eval_dataset_names']}")
        eval_dataset = JSONLDataset(self.args["eval_dataset_names"], tuner=self, args=self.args,)

        match self.fine_tuning:
            case "freeze": # 暂未支持，仅打印模型结构
                from torchinfo import summary
                print(f"Freeze model parameters")
                print(model)
                summary(model=model)
                for name, param in model.named_parameters():
                    print(f"{name}: requires_grad={param.requires_grad}")
            case "lora":
                target_modules = ModelUtil.find_all_linear_modules(model, self.args.get("freeze_vision_tower", True))
                lora_config = LoraConfig(
                    r=self.args.get("lora_rank", 8),  # 低秩矩阵的秩
                    lora_alpha=self.args.get("lora_alpha", 8),  # 缩放因子
                    target_modules=target_modules,  # 目标模块
                    lora_dropout=self.args.get("lora_dropout", 0.0),
                )
                print(f"Patch lora config: {lora_config}")
                # 应用LoRA到模型
                model = get_peft_model(model, lora_config)
        print(f"Train...")
        trainer = self.trainer(model, train_dataset=train_dataset, eval_dataset=eval_dataset,)
        if trainer.args.do_train:
            result = trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)
            trainer.log_metrics("train", result.metrics)
            trainer.save_metrics("train", result.metrics)
            trainer.save_state()
            if trainer.is_world_process_zero():
                PlotUtil.plot_loss(trainer.args.output_dir, keys=["loss", "eval_loss"])
        print(f"Evaluate...")
        if trainer.args.do_eval:
            metrics = trainer.evaluate(metric_key_prefix="eval")
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        print(f"Save model to {self.output_dir}")
        trainer.save_model()
        print(f'Done.')

    def model(self):
        match self.quantization_method:
            case "bitsandbytes":
                match self.quantization_bit:
                    case 4:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=self.args.get("bnb_4bit_compute_dtype"),
                            bnb_4bit_use_double_quant=self.args.get("bnb_4bit_use_double_quant", False),
                            bnb_4bit_quant_type=self.args.get("bnb_4bit_quant_type", "fp4"),
                            bnb_4bit_quant_storage=self.args.get("bnb_4bit_quant_storage"),
                        )
                    case 8:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                    case _:
                        raise Exception(
                            f"Bitsandbytes only accepts 4-bit or 8-bit quantization, but got {self.quantization_bit}")
                        # 加载量化模型
                print(f"Quantization config: {quantization_config}")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    quantization_config=quantization_config,
                    device_map=self.args.get("device_map", "auto"),
                )
            case _:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map=self.args.get("device_map", "auto"),
                )
        return model

    @abstractmethod
    def trainer(
            self,
            model,
            train_dataset: Optional[Union[JSONLDataset, JSONLStreamingDataset]] = None,
            eval_dataset: Optional[Union[JSONLDataset, JSONLStreamingDataset]] = None,
    ) -> Trainer:
        pass

    @abstractmethod
    def sample(self, dataset: DatasetConfig, line: str) -> List[Any]:
        pass

def main(args: Dict):
    match args.get("stage"):
        case "pt":
            from fs_gpt.train.pt.PtTuner import PtTuner
            tuner = PtTuner(args)
            pass
        case "sft":
            from fs_gpt.train.sft.SftTuner import SftTuner
            tuner = SftTuner(args)
            pass
        case "rlhf":
            from fs_gpt.train.rlhf.RlhfTuner import RlhfTuner
            tuner = RlhfTuner(args)
        case _:
            raise Exception(f"Unknown stage: {args.get('stage')}")
    tuner.train()

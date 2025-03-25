from typing import Dict

from transformers import AutoTokenizer

class Derive:
    def __init__(self, args: Dict) -> None:
        self.args = args
        self.device = args.get("device")
        self.derive_method = args.get("derive_method")
        self.derive_dir = args.get("derive_dir")
        self.derive_size = args.get("derive_size", 4)
        self.model_name_or_path = args.get("model_name_or_path")
        self.quantization_bit = args.get("quantization_bit", 4)

    def generate(self):
        match self.derive_method:
            case 'gptq':
                self.gptq()
            case 'awq':
                self.awq()
            case _:
                print(f"Unknown derive_method: {self.derive_method}")

    def gptq(self):
        from gptqmodel import GPTQModel, QuantizeConfig

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        examples = [
            tokenizer(
                "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]
        quantize_config = QuantizeConfig(
            bits=self.quantization_bit,  # quantize model to 4-bit
            group_size=128,  # it is recommended to set the value to 128
            device=self.device,
        )

        print(f"Load model from {self.model_name_or_path}")
        model = GPTQModel.load(self.model_name_or_path, quantize_config=quantize_config,)
        print(f"Quantize with {len(examples)} examples")
        model.quantize(examples)
        print(f"Save model to {self.derive_dir}")
        model.save(self.derive_dir, max_shard_size=f"{self.derive_size}GB",)
        tokenizer.save_pretrained(self.derive_dir)
        print(f'Model is quantized and saved at "{self.derive_dir}"')

    def awq(self):
        from awq import AutoAWQForCausalLM

        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": self.quantization_bit
        }

        print(f"Load model from {self.model_name_or_path}")
        model = AutoAWQForCausalLM.from_pretrained(self.model_name_or_path, device_map=self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)

        print(f"Quantize with config {quant_config}")
        model.quantize(tokenizer, quant_config=quant_config)

        print(f"Save model to {self.derive_dir}")
        model.save_quantized(self.derive_dir, shard_size=f"{self.derive_size}GB",)
        tokenizer.save_pretrained(self.derive_dir)
        print(f'Model is quantized and saved at "{self.derive_dir}"')

def main(args: Dict):
    Derive(args).generate()

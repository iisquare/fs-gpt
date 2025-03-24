from typing import Dict

from transformers import AutoTokenizer

class Derive:
    def __init__(self, args: Dict) -> None:
        self.args = args
        self.device = args.get("device")
        self.derive_method = args.get("derive_method")
        self.derive_dir = args.get("derive_dir")
        self.derive_size = args.get("derive_size", 4)

    def generate(self):
        match self.args['derive_method']:
            case 'gptq':
                self.gptq()
            case _:
                print(f"Unknown derive_method: {self.args['derive_method']}")

    def gptq(self):
        from gptqmodel import GPTQModel, QuantizeConfig

        tokenizer = AutoTokenizer.from_pretrained(self.args["model_name_or_path"], use_fast=True)
        examples = [
            tokenizer(
                "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]

        quantize_config = QuantizeConfig(
            bits=self.args.get("quantization_bit", 4),  # quantize model to 4-bit
            group_size=128,  # it is recommended to set the value to 128
            device=self.device,
        )

        print('load model...')

        model = GPTQModel.load(self.args["model_name_or_path"], quantize_config=quantize_config,)

        print('quantize...')

        model.quantize(examples)

        print('save...')

        model.save(self.derive_dir, max_shard_size=f"{self.derive_size}GB",)

        tokenizer.save_pretrained(self.derive_dir)

        print('done.')

def main(args: Dict):
    Derive(args).generate()

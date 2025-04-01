import os
from pathlib import Path
from typing import Dict

import sentencepiece
from transformers import AutoTokenizer


class Vocab:
    def __init__(self, args: Dict):
        self.args = args
        self.stage = args.get("stage")
        self.output_dir = args.get("output_dir", f"logs/vocab")

    def train(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        model_prefix = str(Path(self.output_dir).joinpath("tokenizer").absolute())
        sentencepiece.SentencePieceTrainer.Train(
            input=self.args.get("input"),
            input_format=self.args.get("input_format", ""),
            model_prefix=model_prefix,
            # 训练语料内容的长度，不能少于词表大小，提示Vocabulary size too high异常
            vocab_size=self.args.get("vocab_size", 512),
            # 指定模型所支持的语言列表，语言代码是ISO639标准定义的缩写，如"en,zh"
            accept_language=self.args.get("accept_language", ""),
            character_coverage=self.args.get("character_coverage", 1.0),
            # UTF-8中一个汉字3个字节，对应.txt一行最多1397个汉字。
            max_sentence_length=self.args.get("max_sentence_length", 4192),
            # 是否将所有数字字符拆分为单独的单元，如”123“拆成”1“，”2“，”3“子词单元
            split_digits=self.args.get("split_digits", True),
            # 在遇到未知或很少的字符时将其分解为 UTF-8 字节来表示，启用后BPE实现的效果就和BBPE一样了
            byte_fallback=self.args.get("byte_fallback", True),
            user_defined_symbols=self.args.get("user_defined_symbols", None),
            model_type=self.args.get("model_type", "bpe"),
        )

    def append(self):
        AutoTokenizer.from_pretrained(self.args.get("model_name_or_path"))
        processor = sentencepiece.SentencePieceProcessor()

    def piece(self):
        match self.stage:
            case "train":
                self.train()
            case "append":
                self.append()
            case "tokenize":
                pass
            case _:
                raise Exception(f"Unknown stage: {self.stage}")

def main(args: Dict):
    Vocab(args).piece()

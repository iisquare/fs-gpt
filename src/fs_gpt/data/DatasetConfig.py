from pathlib import Path
from typing import Dict, Any

import yaml

"""
配置格式：
[name]:
    filepath: /path/to/dataset,
    text_field: content,
"""
class DatasetConfig:
    def __init__(self, args: Dict):
        self.args = args
        self.dataset = yaml.safe_load(Path(args.get("dataset_config", "config/dataset.yaml")).read_text())
        self.block_size = args.get("dataset_block_size", 1024)
        self.overlap = args.get("dataset_overlap", 0)
        assert self.overlap < self.block_size, "Overlap must be less than block size"

    def path(self, name: str) -> str:
        return str(Path(self.dataset[name]["filepath"]).absolute())

    def text(self, name: str, line: str) -> Any | None:
        data = yaml.safe_load(line)
        if not data:
            return None
        return data.get(self.dataset[name]["text_field"])

    def split(self, sample):
        result = []
        start = 0
        end = self.block_size
        length = len(sample)
        while start < length:
            chunk = sample[start:end]
            result.append(chunk)
            start += self.block_size - self.overlap
            end = start + self.block_size
        return result

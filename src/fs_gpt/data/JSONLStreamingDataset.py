import re
from typing import Dict, List, Union

import torch
from torch.utils.data import IterableDataset

from fs_gpt.data.DatasetConfig import DatasetConfig


class JSONLStreamingDataset(IterableDataset):

    def __init__(self, dataset_names: Union[str, List[str]], tokenizer, args: Dict):
        self.dataset_names = dataset_names
        if isinstance(dataset_names, str):
            self.dataset_names = [name.strip() for name in re.split(r'[,;\s]+', dataset_names) if name.strip()]
        self.tokenizer = tokenizer
        self.dataset = DatasetConfig(args)


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset_names = self._split_files(worker_info)

        for name in dataset_names:
            with open(self.dataset.path(name), 'r', encoding='utf-8') as f:
                for line in f:
                    line = self.dataset.text(name, line.strip())
                    if not line:
                        continue
                    for sample in self.dataset.split(line):
                        encoding = self.tokenizer(sample, return_tensors="pt",)
                        # 去掉 batch 维度
                        input_ids = encoding["input_ids"].squeeze(0)
                        attention_mask = encoding["attention_mask"].squeeze(0)
                        yield {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "labels": input_ids
                        }

    def _split_files(self, worker_info):
        """分配文件给不同的worker"""
        if worker_info is None:
            return self.dataset_names

        per_worker = len(self.dataset_names) // worker_info.num_workers
        worker_id = worker_info.id
        start = worker_id * per_worker
        end = start + per_worker if worker_id < worker_info.num_workers - 1 else None
        return self.dataset_names[start:end]

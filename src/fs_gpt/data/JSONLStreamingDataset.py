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

        buffer = []
        for name in dataset_names:
            with open(self.dataset.path(name), 'r', encoding='utf-8') as f:
                for line in f:
                    line = self.dataset.text(name, line.strip())
                    if not line:
                        continue
                    # Tokenize and add to buffer
                    tokens = self.tokenizer.encode(line, add_special_tokens=False)
                    buffer.extend(tokens)
                    # Yield blocks while buffer is sufficient
                    while len(buffer) >= self.dataset.block_size:
                        yield torch.tensor(buffer[:self.dataset.block_size], torch.long)
                        buffer = buffer[self.dataset.block_size - self.dataset.overlap:]

    def _split_files(self, worker_info):
        """分配文件给不同的worker"""
        if worker_info is None:
            return self.dataset_names

        per_worker = len(self.dataset_names) // worker_info.num_workers
        worker_id = worker_info.id
        start = worker_id * per_worker
        end = start + per_worker if worker_id < worker_info.num_workers - 1 else None
        return self.dataset_names[start:end]

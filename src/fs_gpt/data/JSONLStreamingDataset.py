from typing import Dict

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import _T_co


class JSONLStreamingDataset(IterableDataset):

    def __getitem__(self, index) -> _T_co:
        pass

    def __init__(self, dataset_names, tokenizer, dataset_config: Dict, arg: Dict):
        self.dataset_names = dataset_names
        self.tokenizer = tokenizer
        self.block_size = arg.get("dataset_block_size", 1024)
        self.overlap = arg.get("dataset_overlap", 0)
        self.dataset_config = dataset_config
        assert self.overlap < self.block_size, "Overlap must be less than block size"

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset_names = self._split_files(worker_info)

        buffer = []
        for name in dataset_names:

            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Tokenize and add to buffer
                    tokens = self.tokenizer.encode(line, add_special_tokens=False)
                    buffer.extend(tokens)

                    # Yield blocks while buffer is sufficient
                    while len(buffer) >= self.block_size:
                        yield torch.tensor(buffer[:self.block_size], torch.long)
                        buffer = buffer[self.block_size - self.overlap:]

    def _split_files(self, worker_info):
        """分配文件给不同的worker"""
        if worker_info is None:
            return self.dataset_names

        per_worker = len(self.dataset_names) // worker_info.num_workers
        worker_id = worker_info.id
        start = worker_id * per_worker
        end = start + per_worker if worker_id < worker_info.num_workers - 1 else None
        return self.dataset_names[start:end]

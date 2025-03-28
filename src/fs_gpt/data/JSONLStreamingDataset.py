import re
from typing import Dict, List, Union, TYPE_CHECKING

import torch
from torch.utils.data import IterableDataset

from fs_gpt.data.DatasetConfig import DatasetConfig

if TYPE_CHECKING:
    from fs_gpt.train.tuner import Tuner


class JSONLStreamingDataset(IterableDataset):

    def __init__(self, dataset_names: Union[str, List[str]], tuner: "Tuner", args: Dict):
        self.args = args
        self.tuner = tuner
        self.dataset_names = dataset_names
        if isinstance(dataset_names, str):
            self.dataset_names = [name.strip() for name in re.split(r'[,;\s]+', dataset_names) if name.strip()]


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset_names = self._split_files(worker_info)
        for name in dataset_names:
            dataset = DatasetConfig(name, self.args)
            with open(dataset.path(), 'r', encoding='utf-8') as f:
                for line in f:
                    samples = self.tuner.sample(dataset, line.strip())
                    for sample in samples:
                        yield sample

    def _split_files(self, worker_info):
        """分配文件给不同的worker"""
        if worker_info is None:
            return self.dataset_names
        per_worker = len(self.dataset_names) // worker_info.num_workers
        worker_id = worker_info.id
        start = worker_id * per_worker
        end = start + per_worker if worker_id < worker_info.num_workers - 1 else None
        return self.dataset_names[start:end]

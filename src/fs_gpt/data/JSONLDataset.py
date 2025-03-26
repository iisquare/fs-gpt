import re
from typing import Dict, List, Union

from torch.utils.data.dataset import Dataset

from fs_gpt.data.DatasetConfig import DatasetConfig


class JSONLDataset(Dataset):
    def __init__(self, dataset_names: Union[str, List[str]], tokenizer, args: Dict):
        self.dataset_names = dataset_names
        if isinstance(dataset_names, str):
            self.dataset_names = [name.strip() for name in re.split(r'[,;\s]+', dataset_names) if name.strip()]
        self.tokenizer = tokenizer
        self.dataset = DatasetConfig(args)
        self.data = []
        self._load_data()

    def _load_data(self):
        for name in self.dataset_names:
            with open(self.dataset.path(name), 'r', encoding='utf-8') as f:
                for line in f:
                    line = self.dataset.text(name, line.strip())
                    if not line:
                        continue
                    tokens = self.tokenizer.encode(line, add_special_tokens=False)
                    self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

import random
from pathlib import Path

import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset


class MAPLEDataset(Dataset):
    def __init__(self, **kwargs):
        self.passages = list()
        self.tokens = list()
        self.labels = list()

        for key, value in kwargs.items():
            setattr(self, key, value)

        assert len(self.passages) == len(self.tokens) == len(self.labels)

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, index):
        return (
            self.passages[index],
            self.tokens[index],
            self.labels[index]
        )

    def load(self, path):
        if not Path(path).exists():
            raise FileNotFoundError

        if path.endswith('.json'):
            load_function = pd.read_json
        elif path.endswith('.csv'):
            load_function = pd.read_csv
        else:
            raise NotImplementedError

        df = load_function(path)
        df = df.drop_duplicates(subset=["passage", "poem"], ignore_index=True)
        passages = df['passage'].tolist()
        tokens = list(map(word_tokenize, passages))
        labels = list()
        for i in range(df.shape[0]):
            indices = df["indices"][i]
            indices_length = len(tokens[i])
            selection_list = torch.zeros(indices_length)
            for idx in indices:
                selection_list[idx] = 1
            labels.append(selection_list)

        assert len(passages) == len(tokens) == len(labels)
        self.passages = passages
        self.tokens = tokens
        self.labels = labels

    def get_batch(self, index, batch_size):
        return (
            self.passages[index:index + batch_size],
            self.tokens[index:index + batch_size],
            self.labels[index:index + batch_size]
        )

    def shuffle(self):
        zipped = list(zip(self.passages, self.tokens, self.labels))
        random.shuffle(zipped)
        self.passages, self.tokens, self.labels = zip(*zipped)
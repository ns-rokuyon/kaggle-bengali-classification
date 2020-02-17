import numpy as np
import random
import tqdm
from collections import defaultdict
from torch.utils.data.sampler import Sampler
from dataset import BengaliSubsetDataset


class PKSampler:
    """Batch sampler
    """
    def __init__(self, dataset, n_iter_per_epoch, p, k=4):
        assert isinstance(dataset, BengaliSubsetDataset)
        self.dataset = dataset
        self.n_iter_per_epoch = n_iter_per_epoch
        self.p = p
        self.k = k

        self.label_to_indices = defaultdict(list)
        for i, g in enumerate(self.dataset.get_grapheme_root_labels()):
            self.label_to_indices[g].append(i)
        self.labels = sorted(list(self.label_to_indices.keys()))
        self.label_counts = [
            len(self.label_to_indices[l])
            for l in self.labels
        ]
        self.label_counts_inv = [
            1 / c
            for c in self.label_counts
        ]
        s = sum(self.label_counts_inv)
        self.label_to_probs = [
            self.label_counts_inv[l] / s
            for l in self.labels
        ]

    @property
    def batch_size(self):
        return self.p * self.k

    def __repr__(self):
        return f'PKSampler(n_iter_per_epoch={self.n_iter_per_epoch}, ' + \
               f'p={self.p}, k={self.k})'

    def __len__(self):
        return self.n_iter_per_epoch

    def __iter__(self):
        batches = self.generate_batch_sequence()
        for batch in batches:
            yield batch

    def generate_batch_sequence(self):
        """Generate batch sequence randomly

        Returns
        -------
        list[list[int]]
        """
        batches = []
        for _ in tqdm.tqdm(range(self.n_iter_per_epoch),
                           desc='Generate batch sequence ...',
                           total=self.n_iter_per_epoch):
            labels = []
            batch = []
            while len(labels) < self.p:
                label = np.random.choice(self.labels,
                                         p=self.label_to_probs)
                if label not in labels:
                    labels.append(label)
            for label in labels:
                batch += random.sample(self.label_to_indices[label],
                                       self.k)
            batches.append(batch)
        return batches
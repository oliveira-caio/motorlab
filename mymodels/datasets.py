import torch


class Deterministic(torch.utils.data.Dataset):
    """
    assumptions:
        - `data`: a dictionary with multiple modalities and modalities are aligned with the same frequency.
        - `intervals`: aligned and at the same frequency of the data.

    how it works:
        accepts a list of valid intervals (eg: `[[100, 2500], [3000, 5000], ...]`), iterates over these intervals and selects 20 frames at a time (by default the frequency is 20Hz, which means i'm loading 1s of data). the stride is used to control how many frames i want to overlap in the data. by default it'll overlap 10 frames, but the general formula for the number of overlapping frames is `seq_length - stride`.

    observation:
        for testing it's better not to have any overlap, so, make `stride = seq_length` when creating the test set.
    """

    def __init__(self, data, intervals, seq_length=20, stride=20):
        self.seq_length = seq_length
        self.stride = stride
        self.intervals = intervals
        self.modalities = list(data.keys())
        self._build_dataset(data)

    def __len__(self):
        return len(self.data[self.modalities[0]])

    def __getitem__(self, idx):
        return {m: torch.tensor(d[idx]) for m, d in self.data.items()}

    def _build_dataset(self, data):
        self.data = dict()
        for m, d in data.items():
            self.data[m] = []
            for s, e in self.intervals:
                for i in range(s, e - self.seq_length + 1, self.stride):
                    # it ignores the end of the sequence if there is no seq_length time points left to extract.
                    if (i + self.seq_length) <= e:
                        self.data[m].append(d[i : i + self.seq_length])

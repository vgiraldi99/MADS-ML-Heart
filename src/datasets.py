from pathlib import Path

import pandas as pd
import torch


class HeartDataset2D:
    def __init__(
        self,
        path: Path,
        target: str,
        shape: tuple[int, int],
    ) -> None:
        df = pd.read_parquet(path)
        self.target = target
        _x = df.drop("target", axis=1)
        x = torch.tensor(_x.values, dtype=torch.float32)

        # original length is 187, which only allows for 11x17 2D tensors
        # 3*2**6 = 192. This makes it easier to reshape the data
        # it also makes convolutions / maxpooling more predictable
        self.x = torch.nn.functional.pad(x, (0, 3 * 2**6 - x.size(1))).reshape(
            -1, 1, *shape
        )
        y = df["target"]
        self.y = torch.tensor(y.values, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def __repr__(self) -> str:
        return f"Heartdataset2D (#{len(self)})"


class HeartDataset1D:
    def __init__(
        self,
        path: Path,
        target: str,
    ) -> None:
        df = pd.read_parquet(path)
        self.target = target
        _x = df.drop("target", axis=1)
        x = torch.tensor(_x.values, dtype=torch.float32)
        # padded to 3*2**6 = 192
        # again, this helps with reshaping for attention & using heads
        self.x = torch.nn.functional.pad(x, (0, 3 * 2**6 - x.size(1)))
        y = df["target"]
        self.y = torch.tensor(y.values, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        # (seq_len, channels)
        return self.x[idx].unsqueeze(1), self.y[idx]

    def __repr__(self) -> str:
        return f"Heartdataset (len {len(self)})"

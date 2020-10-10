from pathlib import Path

import typer
import pandas as pd
import numpy as np


def main(path: str, n: int = 1):
    base_dir = Path(path)
    val_dir = base_dir / ".." / "valid"
    val_dir.mkdir(exist_ok=True)
    counts = []
    for filepath in base_dir.iterdir():
        if filepath.is_dir():
            examples = list(filepath.iterdir())
            counts.append(len(examples))
            to_be_moved = np.random.choice(examples, size=n, replace=False)
            target_dir = val_dir / filepath.name
            target_dir.mkdir(exist_ok=True)
            for item in to_be_moved:
                item.rename(target_dir / item.name)
    counts_series = pd.Series(counts)
    print(counts_series.describe())


if __name__ == "__main__":
    typer.run(main)

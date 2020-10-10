from pathlib import Path

import typer


def main(path: str):
    base_dir = Path(path)
    for filepath in base_dir.iterdir():
        shards = filepath.name.split("_")
        age = shards[0]
        idx = shards[-1].split(".")[0]
        name = "_".join(shards[1:-1])
        target_dir = base_dir / "train" / name
        target_dir.mkdir(exist_ok=True)
        filepath.rename(target_dir / f"{age}_{idx}.jpg")
    print(f"# of people: {len(list(base_dir.iterdir()))}")


if __name__ == "__main__":
    typer.run(main)

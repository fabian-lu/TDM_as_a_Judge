"""Reading the data files. Large files are stored gzip-compressed (`name.jsonl.gz`) and read as if they were not."""
import gzip
from pathlib import Path


def find(path):
    """`path`, or `path.gz` if only the compressed file exists, or None if neither exists."""
    path = Path(path)
    if path.exists():
        return path
    gz = path.with_name(path.name + ".gz")
    return gz if gz.exists() else None


def read_lines(path):
    """The lines of a text file, compressed or not ([] if it does not exist)."""
    found = find(path)
    if found is None:
        return []
    opener = gzip.open if found.suffix == ".gz" else open
    with opener(found, "rt", encoding="utf-8") as f:
        return f.read().splitlines()

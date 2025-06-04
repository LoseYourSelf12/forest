from typing import Iterable, Tuple
import pandas as pd


class Dataset:
    def __init__(self, csv_path: str, img_col: str = "file_name", label_col: str = "category"):
        self.df = pd.read_csv(csv_path)
        self.img_col = img_col
        self.label_col = label_col

    def __iter__(self) -> Iterable[Tuple[str, str]]:
        for _, row in self.df.iterrows():
            yield row[self.img_col], row[self.label_col]

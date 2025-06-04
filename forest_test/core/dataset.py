from typing import Iterable, Tuple, List
import pandas as pd
import re

class Dataset:
    def __init__(self, csv_path: str,
                 img_col: str = "file_name",
                 label_col: str = "category",
                 current_col: str = "current_file_name",
                 ds_col: str = "dataset"):
        self.df = pd.read_csv(csv_path)
        self.img_col = img_col
        self.label_col = label_col
        self.current_col = current_col
        self.ds_col = ds_col

    def parse_categories(self, cat_str: str) -> List[str]:
        if not isinstance(cat_str, str):
            return []
        ids = list(re.finditer(r"\d{2}\.\d{2}", cat_str))
        if not ids:
            return [cat_str.strip()] if cat_str else []
        categories = []
        for i, m in enumerate(ids):
            start = m.start()
            end = ids[i + 1].start() if i + 1 < len(ids) else len(cat_str)
            categories.append(cat_str[start:end].strip().strip(','))
        return categories

    def __iter__(self) -> Iterable[Tuple[str, List[str], str, str]]:
        for _, row in self.df.iterrows():
            yield (
                row[self.img_col],
                self.parse_categories(row[self.label_col]),
                row.get(self.current_col, row[self.img_col]),
                row.get(self.ds_col, ""),
            )

    def __len__(self) -> int:
        return len(self.df)

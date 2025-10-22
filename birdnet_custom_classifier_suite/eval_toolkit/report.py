
from typing import List
import pandas as pd

# Why: Keep I/O tiny and explicit.
def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def render_markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    view = df.head(max_rows)
    return view.to_markdown(index=False)

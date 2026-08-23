# save_dataset.py (run once, then delete or keep)
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris(as_frame=True)
df = data.frame
df.to_csv("data/dataset.csv", index=False)
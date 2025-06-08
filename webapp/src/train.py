import os
import pandas as pd
from model import train_model

DATA_PATH = os.path.join("data", "raw", "train.csv")

def train():
    df = pd.read_csv(DATA_PATH)

    # You must explicitly specify algorithm name now
    train_model(df, algorithm_name="random_forest")
    train_model(df, algorithm_name="gradient_boosting")

if __name__ == "__main__":
    train()

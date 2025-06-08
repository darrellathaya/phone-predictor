import os
import pandas as pd
from model import train_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw", "train.csv")

def train():
    df = pd.read_csv(DATA_DIR)

    # You must explicitly specify algorithm name now
    train_model(df, algorithm_name="random_forest")
    train_model(df, algorithm_name="gradient_boosting")

if __name__ == "__main__":
    train()

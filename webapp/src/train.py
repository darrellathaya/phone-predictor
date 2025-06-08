import os
import pandas as pd
from model import train_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "data/raw")
META_PATH = os.path.join(MODEL_DIR, "train.csv")

def train():
    df = pd.read_csv(DATA_PATH)

    # You must explicitly specify algorithm name now
    train_model(df, algorithm_name="random_forest")
    train_model(df, algorithm_name="gradient_boosting")

if __name__ == "__main__":
    train()

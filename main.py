import pandas as pd
from model import RNNModel

def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    print(f"Data loaded from {csv_file}")
    return df

csv_file = '30m.csv'

df = load_data_from_csv(csv_file)

model = RNNModel(forecast_horizon=3, timestep=5, df=df, split=0.8)



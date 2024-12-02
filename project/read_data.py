import pandas as pd
import numpy as np

def obtain_dataset():
    df = pd.read_csv('project/bike_dataset.csv')
    return df

obtain_dataset()
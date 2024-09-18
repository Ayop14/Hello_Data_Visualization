import pandas as pd
import numpy as np

def obtain_dataset():
    df = pd.read_csv('bike_dataset.csv', dtype={
        'brand': str,
        'name': str,
        'consumption': np.float32,
        'accelerarion': np.float32,
        'max speed': np.float32,
        'bike type': np.float32,
        'power': np.float32,
        'rpm max power': np.float32,
        'cylinders': np.int16,
        'refrigeration': str,
        'torque': np.float32,
        'weight full': np.float32,
        'seat height': np.float32,
        'gasoline capacity': np.float32,
        'price': np.float32,
        'number of gears': np.float32,
        'brake pistons': np.float32,
        'front brake disk size': np.float32,
        'back brake disk size': np.float32,
        'length': np.float32,
        'width': np.float32,
        'height': np.float32
    })


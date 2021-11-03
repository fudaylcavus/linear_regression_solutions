import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
house_data = pd.read_csv('streeteasy.csv')
x_all = house_data[['size_sqft', 'min_to_subway']]
y_all = house_data.rent  

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all)
    

def get_execution_time_by_running(func, **kwargs):
    start = time.time()
    output = func(**kwargs)
    end = time.time()
    print('Runtime: ', round((end - start), 2), 's')
    return output

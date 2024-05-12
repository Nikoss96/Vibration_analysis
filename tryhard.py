import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import os
from models import *

"""
def prepair_test(file_path = "C:/Users/nikch/Vibrations_analysis/0_check.csv"):
    data = pd.read_csv(file_path)
    X.append(data[['X', "Y", "Z"]].values)
    y.append(0)
    X = np.array(X)
    y = np.array(y)
    
    y = to_categorical(y, num_classes=10)
    return X , y
"""
def load_data():
    X, y = [], []
    for label in range(10):
        path = f'data_membr/{label}/'
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path)
            X.append(data[['X', "Y", "Z"]].values)
            y.append(label)
    X = np.array(X)
    y = np.array(y)

    y = to_categorical(y, num_classes=10)
    return train_test_split(X, y, test_size=0.3, random_state=15)

X_train, X_test, y_train, y_test = load_data()

#1
#lstmgruconv(X_train, X_test, y_train, y_test)


#2
#lstm(X_train, X_test, y_train, y_test)

#3
check1(X_train, X_test, y_train, y_test)

#4 Gru + lstm + conv + dense mix






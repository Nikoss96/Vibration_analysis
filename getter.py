import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

media_path = "data\\"

def read_csv_files_with_labels(root_folder):
    data = []
    for i in range(10):
        folder_path = os.path.join(root_folder, str(i))
        if not os.path.exists(folder_path):
            continue
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        for file in files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            df['label'] = i
            data.append(df)
    return pd.concat(data, ignore_index=True)

#df_all = read_csv_files_with_labels(media_path)
#print(df_all)


def prepair_test(file_path = "C:/Users/nikch/Vibrations_analysis/4_check.csv"):
    X, y = [], []
    data = pd.read_csv(file_path)
    X.append(data[['X', "Y", "Z"]].values)
    y.append(0)
    X = np.array(X)
    y = np.array(y)
    
    y = to_categorical(y, num_classes=10)
    return X


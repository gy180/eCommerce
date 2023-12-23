import numpy as np 
import pandas as pd 
import torch

# remove outliers
def remove_outliers(dataset, idx_set):
    """
    remove outliers from a dataset

    Input: 
        -data: an array representing the dataset
        -idx_set: a set representing the indices set from previous iterations

    Returns a set of indices that represent what needs to be removed
    """
    outlier_idx = idx_set
    mean_val = np.mean(dataset)
    std_dev = np.std(dataset)
    
    #find the indices (it is an outlier when it is over 3 standard dev from mean)
    for idx in range(0,len(dataset)):
        if mean_val - 3*std_dev > dataset[idx] or dataset[idx] > mean_val + 3*std_dev:
            outlier_idx.add(idx)

    return outlier_idx

# reading/extracting data from the file
df = pd.read_csv('data/creditcard.csv')

# find all the outliers
headers = list(df.columns)
outliers = set()
for idx in range(1,len(headers)-2):
    header = headers[idx]
    outliers = remove_outliers(df[header], outliers)

# remove the rows that include the outliers
filtered = df.values.tolist()
for idx in outliers:
    filtered.pop(idx)

fraud = []
not_fraud = []
# separate the fraud and non fraud data
for idx in range(0, len(df['Class'])):
    if filtered[idx][-1] == 0:
        not_fraud.append(filtered[idx])
    else:
        fraud.append(filtered[idx])

#
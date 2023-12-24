import numpy as np 
import pandas as pd 
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

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

class CreditCardDataSet(Dataset):
    """Credit Card Fraud dataset"""
    def __init__(self, fraud, not_fraud):
        self.fraud = fraud
        self.not_fraud = not_fraud
    
    def __len__ (self):
        """returns the length of the dataset"""
        return len(self.fraud) + len(self.not_fraud)
    
    def __getitem__(self, index):
        """returns a random row in df"""
        rnd = random.randint(0, 3)
        rnd_lst = self.not_fraud

        # make data more balanced
        if rnd == 0:
            rnd_lst = self.fraud
            upper = int(len(self.fraud) *0.75)
        else:
            upper = int(len(self.not_fraud)*0.75)

        rnd_idx = random.randint(0,upper)

        return torch.tensor(rnd_lst.iloc[rnd_idx].values.astype(np.float32))
    
class FraudDetectionModel(nn.Module):
    """model that the machine is trained on"""
    def __init__(self, input_size, hidden_size, output_size):
        super(FraudDetectionModel,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """does the computation"""
        return self.model(x)


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
idx_outliers = idx_outliers = sorted(outliers, reverse=True)
for idx in idx_outliers:
    filtered.pop(idx)

# separate the fraud and non fraud data
fraud = []
not_fraud = []
for idx in range(0, len(df['Class'])-len(idx_outliers)):
    if filtered[idx][-1] == 0:
        not_fraud.append(filtered[idx])
    else:
        fraud.append(filtered[idx])

fraud_df = pd.DataFrame(fraud, columns=headers)
not_fraud_df = pd.DataFrame(not_fraud, columns=headers)

# split into training and testing data sets
# train_size = 0.75
# fraud_train, fraud_test = train_test_split(fraud_df, train_size=train_size, shuffle=True)
# not_fraud_train, not_fraud_test = train_test_split(fraud_df, train_size=train_size, shuffle=True)

# create DataSets
train_dataset = CreditCardDataSet(fraud_df, not_fraud_df)

# create DataLoaders
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# creating the neural network model
input_size = len(df.columns) - 1
output_size = 1
hidden_size = int(input_size*2/3) + output_size
model = FraudDetectionModel(input_size, hidden_size, output_size)

# optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_func = nn.BCELoss()

# training loop
epoch_num = 50
for epoch in range(0, epoch_num):
    model.train()
    for transaction in train_dataloader:
        input = transaction[:, :-1]
        actual = transaction[:, -1]

        optimizer.zero_grad()
        output = model(input)
        loss = loss_func(output, actual)
        loss.backward()
        optimizer.step()

    #validation
    model.eval()
    
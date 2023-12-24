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
    # mean_val = np.mean(dataset)
    # std_dev = np.std(dataset)
    # lower_bound, upper_bound = mean_val - 3*std_dev, mean_val + 3*std_dev

    #box and whiskers 
    quartile25, quartile75 = np.percentile(dataset, 25), np.percentile(dataset,75)
    interquart_range = quartile75 - quartile25
    cutoff = interquart_range*1.75
    lower_bound, upper_bound = quartile25 - cutoff, quartile75+cutoff

    #iterate through data set and find the outliers
    for idx in range(0,len(dataset)):
        if lower_bound > dataset[idx] or dataset[idx] > upper_bound:
            outlier_idx.add(idx)
    
    return outlier_idx

def validate(set, model):
    """
    check if the model is training and if its improving
    Input:
        -set: represents the validation set
        -model: the trained model
    Returns the ratio of correct predictions over all predictions
    """
    input = set[:, :-1]
    actual = set[:, -1].tolist()
    predicted = model(input).view(-1).tolist()
    rounded_predicted = [float(round(pred)) for pred in predicted]
    print("actual", actual)
    print("predicted", rounded_predicted)

    correct = 0
    for pred, act in zip(actual, rounded_predicted):
        if pred == act:
            correct += 1
    
    ratio = correct/len(set)
    return ratio

class CreditCardDataSet(Dataset):
    """Credit Card Fraud dataset"""
    def __init__(self, fraud, not_fraud, batch_len):
        """
        Initialize the dataset
        Input:
            -fraud: dataframe representing the fraud data
            -not_fraud: dataframe rerpesenting the data that aren't classified as fraud
        """
        self.fraud = fraud
        self.not_fraud = not_fraud
        self.batch_len = batch_len
    
    def __len__ (self):
        """returns the length of the dataset"""
        total_len = len(self.fraud) + len(self.not_fraud)
        return  total_len - (total_len% self.batch_len)
    
    def __getitem__(self, index):
        """returns a random row in df"""
        rnd = random.randint(0, 1)
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
        """
        Initalize the fraud detection model

        Input:
            -input_size: an int, which is the number of input features
            -hidden_size: an int representing num of units in the hidden feature
            -output_size: an int representing the number of output features
        """
        super(FraudDetectionModel,self).__init__()
        hidden_size2 = int(hidden_size*2/3) + output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
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
idx_outliers = sorted(outliers, reverse=True)
for idx in idx_outliers:
    filtered.pop(idx)

# separate the fraud and non fraud data
fraud = []
not_fraud = []
for idx in range(0, len(filtered)):
    if filtered[idx][-1] == 0:
        not_fraud.append(filtered[idx])
    else:
        fraud.append(filtered[idx])

fraud_df = pd.DataFrame(fraud, columns=headers)
not_fraud_df = pd.DataFrame(not_fraud, columns=headers)

# create DataSets and DataLoaders (training)
batch_size = 64
train_dataset = CreditCardDataSet(fraud_df, not_fraud_df, batch_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# creating the neural network model
input_size = len(df.columns) - 1
output_size = 1
hidden_size = int(input_size*3/4) + output_size
model = FraudDetectionModel(input_size, hidden_size, output_size)

# optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.BCELoss()

# creating the validation set
# validation_set = torch.tensor(fraud[0:401] + not_fraud[0:401])
validation_set = torch.tensor(fraud[0:401] + not_fraud[0:401], dtype=torch.float32)
print(validation_set.shape)

# training loop
epoch_num = 20
model.train()
for epoch in range(0, epoch_num):
    for transaction in train_dataloader:
        input = transaction[:, :-1]
        actual = transaction[:,-1]

        # optimize and backpropogate
        optimizer.zero_grad()
        output = model(input)
        loss = loss_func(output, torch.reshape(actual,(64,1)))
        loss.backward()
        optimizer.step()

    #validate the data
    if (epoch + 1)%10 == 0:
        print(validate(validation_set, model))


# test_set = torch.tensor(fraud[int(len(fraud)*0.75):] + not_fraud[int(len(not_fraud*0.75)):])
# print(validate(test_set, model))
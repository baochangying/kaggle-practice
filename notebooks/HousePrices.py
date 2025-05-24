import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

RAW_DATA = os.path.join('..', 'data', 'raw', 'house-prices-advanced-regression-techniques')

train_df = pd.read_csv(os.path.join(RAW_DATA, 'train.csv'))
test_df = pd.read_csv(os.path.join(RAW_DATA, 'test.csv'))

all_features = pd.concat((train_df.iloc [:, 1:-1], test_df.iloc [:, 1:]))

numeric_cols = all_features.select_dtypes(exclude=['object']).columns

all_features[numeric_cols] = all_features[numeric_cols].apply(
    lambda col: (col - col.mean()) / col.std()
)
all_features[numeric_cols] = all_features[numeric_cols].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape

n_train = train_df.shape[0]
all_features = all_features.astype(float)
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_df.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

loss = torch.nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = torch.nn.Sequential(torch.nn.Linear(in_features, 1))
    return net

def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    
    train_dataset = TensorDataset(train_features, train_labels.float())
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y.float())
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels.float()))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels.float()))          
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y) :
    assert k > 1
    fold_size = X. shape [0] // k
    X_train, Y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, Y_part = X[idx, :], y[idx]
        if j == i:  
            X_valid, Y_valid = X_part, Y_part
        elif X_train is None:
            X_train, Y_train = X_part, Y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            Y_train = torch.cat([Y_train, Y_part], 0)
    return X_train, Y_train, X_valid, Y_valid

def k_fold(k,X_train, Y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0 
    for i in range (k):
        data = get_k_fold_data(k, i, X_train, Y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0: 
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,4))
            plt.plot(range(1, num_epochs+1), train_ls, label='train')
            plt.plot(range(1, num_epochs+1), valid_ls, label='valid')
            plt.xlabel('epoch')
            plt.ylabel('log rmse')
            plt.xlim([1, num_epochs])
            plt.legend()
            plt.show()
        print(f'fold {i+1}, train log rmse {train_ls[-1]:f}, valid log rmse {valid_ls[-1]:f}')
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs,
                          lr,weight_decay, batch_size) 

print(f"{k}-折验证：平均训练logrmse：{float(train_l):f},"
      f"平均验证logrmse：{float(valid_l):f}")

def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    
    print(f'train log rmse {float(train_ls[-1]) :f}')
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape (1, - 1) [0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
train_and_pred(train_features, test_features, train_labels, test_df,num_epochs, lr, weight_decay, batch_size)   
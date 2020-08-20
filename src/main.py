import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SKLearnLogisticRegression
from sklearn.base import clone
from scipy import stats
import math
import Orange
from skorch import NeuralNetClassifier
import itertools
import random


class LogisticRegressionSoftmax(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionSoftmax, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        return torch.nn.functional.softmax(self.fc(x), dim=1)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        return self.fc(x)


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.ReLU = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        hidden = self.fc1(x)
        ReLU = self.ReLU(hidden)
        output = self.fc2(ReLU)
        return output


def friedman_test(*args, reverse=False):

    k = len(args)
    if k < 2:
        raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1:
        raise ValueError('Unequal number of samples')

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row, reverse=reverse)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2. for v in row])

    rankings_avg = [np.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r / np.sqrt(k * (k + 1) / (6. * n)) for r in rankings_avg]

    chi2 = ((12 * n) / float((k * (k + 1)))) * ((np.sum(r**2 for r in rankings_avg)) - ((k * (k + 1)**2) / float(4)))
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - stats.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def wilcoxon_test(score_A, score_B):

    # compute abs delta and sign
    delta_score = [score_B[i] - score_A[i] for i in range(len(score_A))]
    sign_delta_score = list(np.sign(delta_score))
    abs_delta_score = list(np.abs(delta_score))

    N_r = float(len(delta_score))

    # hadling scores
    score_df = pd.DataFrame({'abs_delta_score': abs_delta_score, 'sign_delta_score': sign_delta_score})

    # sort
    score_df.sort_values(by='abs_delta_score', inplace=True)
    score_df.index = range(1, len(score_df) + 1)

    # adding ranks
    score_df['Ranks'] = score_df.index
    score_df['Ranks'] = score_df['Ranks'].astype('float64')

    score_df.dropna(inplace=True)

    # z : pouput value
    W = sum(score_df['sign_delta_score'] * score_df['Ranks'])
    z = W / (math.sqrt(N_r * (N_r + 1) * (2 * N_r + 1) / 6.0))

    # rejecte or not the null hypothesis
    null_hypothesis_rejected = False
    if z < -1.96 or z > 1.96:
        null_hypothesis_rejected = True

    return z, null_hypothesis_rejected


def noisy_completly_at_random(y, ratio):
    n = y.shape

    is_missing = np.random.binomial(1, ratio, n)
    missing_value = np.random.binomial(1, 0.5, len(y[is_missing == 1]))

    y_missing = np.copy(y)
    y_missing[is_missing == 1] = missing_value

    return y_missing


def noisy_not_at_random(proba, y, ratio):

    n = y.shape

    if ratio == 1:
        scaled = np.full_like(proba, 1)
    else:
        scaled = 1-(1-ratio)*np.power(np.abs(1-2*proba), 1/(1-ratio))
    is_missing = np.random.binomial(1, scaled, n)
    missing_value = np.random.binomial(1, 0.5, len(y[is_missing == 1]))

    y_missing = np.copy(y)
    y_missing[is_missing == 1] = missing_value

    return y_missing


def split_dataset(dataset, split):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=split, stratify=dataset[:][1])
    train = torch.utils.data.Subset(dataset, train_idx)
    val = torch.utils.data.Subset(dataset, val_idx)
    return (XYDataset(train[:][0], train[:][1]), XYDataset(val[:][0], val[:][1]))


def split_scale_dataset(dataset, split):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=split, stratify=dataset[:][1])
    train = torch.utils.data.Subset(dataset, train_idx)
    val = torch.utils.data.Subset(dataset, val_idx)
    scaler = StandardScaler().fit(train[:][0])
    scaled_train = XYDataset(scaler.transform(train[:][0]), train[:][1])
    if val[:][0].shape[0] == 0:
        scaled_val = val
    else:
        scaled_val = XYDataset(scaler.transform(val[:][0]), val[:][1])
    return (scaled_train, scaled_val)


def corrupt_dataset(dataset, corrupt_fn, cr):
    return XYDataset(dataset[:][0], corrupt_fn(dataset[:][1], cr))


def split_corrupt_dataset(dataset, corrupt_fn, split, cr):
    trusted_idx, untrusted_idx = train_test_split(list(range(len(dataset))), test_size=split, stratify=dataset[:][1])
    trusted = torch.utils.data.Subset(dataset, trusted_idx)
    untrusted = torch.utils.data.Subset(dataset, untrusted_idx)
    corrupted = XYDataset(untrusted[:][0], corrupt_fn(untrusted[:][1], cr))
    return (XYDataset(trusted[:][0], trusted[:][1]), corrupted)


class UnhingedLoss(torch.nn.Module):
    def __init__(self):
        super(UnhingedLoss, self).__init__()

    def forward(self, X, y):
        return 1 - (2*y-1) * X[:, 1]


class ad(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (pd.read_csv("data/ad/train.csv").iloc[:, :-1].to_numpy(),
                (pd.read_csv("data/ad/train.csv").iloc[:, -1].to_numpy() == 'ad.').astype(int))
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)



class web(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (datasets.load_svmlight_file("data/web/train")[0].todense(), datasets.load_svmlight_file("data/web/train")[1])
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class fourclass(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (datasets.load_svmlight_file("data/fourclass/train")[0].todense(), datasets.load_svmlight_file("data/fourclass/train")[1])
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)

class svmguide3(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (datasets.load_svmlight_file("data/svmguide3/train")[0].todense(), datasets.load_svmlight_file("data/svmguide3/train")[1])
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class svmguide1(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (datasets.load_svmlight_file("data/svmguide1/train")[0].todense(), datasets.load_svmlight_file("data/svmguide1/train")[1])
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class ionosphere(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (pd.read_csv("data/ionosphere/train.csv").iloc[:, :-1].to_numpy(),
                (pd.read_csv("data/ionosphere/train.csv").iloc[:, -1].to_numpy() == "b").astype(int))
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class banknote(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (pd.read_csv("data/banknote/train.csv", header=None).iloc[:, :-1].to_numpy(),
                (pd.read_csv("data/banknote/train.csv", header=None).iloc[:, -1].to_numpy()).astype(int))
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class musk(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (pd.read_csv("data/musk/train.csv", header=None).iloc[:, 2:-1].to_numpy(),
                (pd.read_csv("data/musk/train.csv", header=None).iloc[:, -1].to_numpy()).astype(int))
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class ijcnn1(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (datasets.load_svmlight_file("data/ijcnn1/train")
                [0].todense(), datasets.load_svmlight_file("data/ijcnn1/train")[1])
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class eeg(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (pd.read_csv("data/eeg/train.csv", header=None).iloc[:, :-1].to_numpy(),
                pd.read_csv("data/eeg/train.csv", header=None).iloc[:, -1].to_numpy())
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class hiva(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (pd.read_csv("data/hiva/train.csv", sep=" ", header=None).iloc[:, :-1].to_numpy(),
                pd.read_csv("data/hiva/label.csv", sep=" ", header=None).iloc[:, 0].to_numpy())
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class ibn_sina(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (pd.read_csv("data/ibn-sina/train.csv", sep=" ", header=None).iloc[:, :-1].to_numpy(),
                pd.read_csv("data/ibn-sina/label.csv", sep=" ", header=None).iloc[:, 0].to_numpy())
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class zebra(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (pd.read_csv("data/zebra/train.csv", sep=" ", header=None).iloc[:, :-1].to_numpy(),
                pd.read_csv("data/zebra/label.csv", sep=" ", header=None).iloc[:, 0].to_numpy())
        self.X, self.y = np.nan_to_num(np.squeeze(np.asarray(X)).astype(
            np.float32), nan=0.0, posinf=0.0, neginf=0.0), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class sylva(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (pd.read_csv("data/sylva/train.csv", sep=" ", header=None).iloc[:, :-1].to_numpy(),
                pd.read_csv("data/sylva/label.csv", sep=" ", header=None).iloc[:, 0].to_numpy())
        self.X, self.y = np.nan_to_num(np.squeeze(np.asarray(X)).astype(
            np.float32), nan=0.0, posinf=0.0, neginf=0.0), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class australian(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (datasets.load_svmlight_file("data/australian/train")
                [0].todense(), datasets.load_svmlight_file("data/australian/train")[1])
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class diabetes(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (datasets.load_svmlight_file("data/diabetes/train")
                [0].todense(), datasets.load_svmlight_file("data/diabetes/train")[1])
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class breast(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (datasets.load_svmlight_file("data/breast-cancer/train")
                [0].todense(), datasets.load_svmlight_file("data/breast-cancer/train")[1])
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class adult(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (datasets.load_svmlight_file("data/adult/train", n_features=123)
                [0].todense(), datasets.load_svmlight_file("data/adult/train", n_features=123)[1])
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class german(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (datasets.load_svmlight_file("data/german/train")
                [0].todense(), datasets.load_svmlight_file("data/german/train")[1])
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class phishing(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (datasets.load_svmlight_file("data/phishing/train")
                [0].todense(), datasets.load_svmlight_file("data/phishing/train")[1])
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class spam(torch.utils.data.Dataset):
    def __init__(self):

        X, y = (pd.read_csv("data/spam/train.csv", header=None).iloc[:, :-1].to_numpy(),
                pd.read_csv("data/spam/train.csv", header=None).iloc[:, -1].to_numpy())
        self.X, self.y = np.squeeze(np.asarray(X)).astype(
            np.float32), LabelEncoder().fit_transform(y).astype(np.long)

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

        pass

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return len(self.X)


class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, trusted, untrusted):
        self.trusted, self.untrusted = trusted, untrusted

        pass

    def __getitem__(self, index):

        if index < len(self.trusted):
            item = self.trusted.__getitem__(index)
            return item[0], (item[1], 0)
        else:
            item = self.untrusted.__getitem__(index - len(self.trusted))
            return item[0], (item[1], 1)

    def __len__(self):

        return len(self.trusted) + len(self.untrusted)


class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, weights):
        self.dataset, self.weights = dataset, weights

        pass

    def __getitem__(self, index):

        return (self.dataset.__getitem__(index)[0], self.weights.__getitem__(index)), self.dataset.__getitem__(index)[1]

    def __len__(self):

        return self.dataset.__len__()


def normal(train, test, optimizer, batch_size, epochs, lr, weight_decay, hidden_size, loss="cross_entropy"):

    input_size = len(train[0][0])
    num_classes = 2

    if hidden_size == 0:
        model = LogisticRegression(input_size, num_classes)
    else:
        model = MLP(input_size, hidden_size, num_classes)

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")

    if loss == "cross_entropy":
        train_loss = torch.nn.CrossEntropyLoss(reduction="none")
    elif loss == "unhinged":
        train_loss = UnhingedLoss()

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1,
                                               drop_last=False)

    valid_loader = torch.utils.data.DataLoader(dataset=test,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1,
                                               drop_last=False)

    mean_train_losses = []
    mean_valid_losses = []
    accs = []

    for epoch in range(epochs):
        model.train()

        train_losses = []
        valid_losses = []

        valid_preds = []
        valid_labels = []

        for i, (data, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs = model(data)

            loss = train_loss(outputs, labels).mean()

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for i, (data, labels) in enumerate(valid_loader):

                outputs = model(data)
                loss = cross_entropy_loss(outputs, labels).mean()

                valid_losses.append(loss.item())

                valid_preds.append(torch.nn.functional.softmax(outputs, dim=1).numpy()[:, 1])
                valid_labels.append(labels.numpy())

            mean_train_losses.append(np.mean(train_losses))
            mean_valid_losses.append(np.mean(valid_losses))

        acc = accuracy_score(np.concatenate(valid_labels), np.concatenate(valid_preds) > 0.5)
        accs.append(acc)

        print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}'
              .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses), acc))

    return model, pd.DataFrame(list(zip(mean_train_losses, mean_valid_losses, accs)),
                               columns=["mean_train_losses", "mean_valid_losses", "accs"])


def irbl(trusted, untrusted, test, ft, fu, optimizer, batch_size, epochs, lr, weight_decay, hidden_size):

    input_size = len(train[0][0])
    num_classes = 2

    if hidden_size == 0:
        model = LogisticRegression(input_size, num_classes)
    else:
        model = MLP(input_size, hidden_size, num_classes)

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
    nll_loss = torch.nn.NLLLoss(reduction="none")

    valid_loader = torch.utils.data.DataLoader(dataset=test,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1,
                                               drop_last=False)

    mean_train_losses = []
    mean_valid_losses = []
    accs = []

    #predict the beta as you go in the last training loop if your data can't fit in memory
    total_data = torch.from_numpy(untrusted[:][0])
    total_labels = torch.from_numpy(untrusted[:][1])

    if hasattr(ft, "predict_proba") and hasattr(fu, "predict_proba"):
        ft_proba = np.take_along_axis(ft.predict_proba(total_data.numpy()),
                                      total_labels.numpy().reshape(-1, 1), axis=1).flatten()
        fu_proba = np.take_along_axis(fu.predict_proba(total_data.numpy()),
                                      total_labels.numpy().reshape(-1, 1), axis=1).flatten()
        beta = np.divide(ft_proba,
                         fu_proba,
                         out=np.zeros_like((total_labels.numpy()), dtype=float),
                         where=fu_proba != 0)
        beta = torch.from_numpy(beta).float()
    else:
        ft_proba = torch.flatten(torch.gather(torch.nn.functional.softmax(
            ft(total_data), dim=1), 1, total_labels.view(-1, 1)))
        fu_proba = torch.flatten(torch.gather(torch.nn.functional.softmax(
            fu(total_data), dim=1), 1, total_labels.view(-1, 1)))
        beta = torch.div(ft_proba,
                         fu_proba)
        beta[torch.isnan(beta)] = 0.0
        beta[torch.isinf(beta)] = 0.0

    total_beta = torch.cat([torch.ones(len(trusted)), beta]).detach()

    total_loader = torch.utils.data.DataLoader(dataset=WeightedDataset(MergedDataset(trusted, untrusted), total_beta),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1,
                                               drop_last=False)

    for epoch in range(epochs):
        model.train()

        train_losses = []
        valid_losses = []

        valid_preds = []
        valid_labels = []

        for i, ((data, weights), (labels, is_corrupteds)) in enumerate(total_loader):

            optimizer.zero_grad()

            outputs = model(data)

            loss = (cross_entropy_loss(outputs, labels) * weights).mean()

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for i, (data, labels) in enumerate(valid_loader):

                outputs = model(data)
                loss = cross_entropy_loss(outputs, labels).mean()

                valid_losses.append(loss.item())

                valid_preds.append(torch.nn.functional.softmax(outputs, dim=1).float().numpy()[:, 1])
                valid_labels.append(labels.numpy())

            mean_train_losses.append(np.mean(train_losses))
            mean_valid_losses.append(np.mean(valid_losses))

        acc = accuracy_score(np.concatenate(valid_labels), np.concatenate(valid_preds) > 0.5)
        accs.append(acc)

        print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}'
              .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses), acc))

    return model, pd.DataFrame(list(zip(mean_train_losses, mean_valid_losses, accs)),
                               columns=["mean_train_losses", "mean_valid_losses", "accs"]), pd.Series(total_beta.detach().numpy())


def glc(trusted, untrusted, test, fu, optimizer, batch_size, epochs, lr, weight_decay, hidden_size):

    input_size = len(trusted[0][0])
    num_classes = int(max(test[:][1]) + 1)

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
    nll_loss = torch.nn.NLLLoss(reduction="none")

    trusted_loader = torch.utils.data.DataLoader(dataset=trusted,
                                                 batch_size=beta_batch_size[0],
                                                 shuffle=True,
                                                 num_workers=1,
                                                 drop_last=False)

    total_loader = torch.utils.data.DataLoader(dataset=MergedDataset(trusted, untrusted),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1,
                                               drop_last=False)

    valid_loader = torch.utils.data.DataLoader(dataset=test,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1,
                                               drop_last=False)

    if hidden_size == 0:
        model = LogisticRegression(input_size, num_classes)
    else:
        model = MLP(input_size, hidden_size, num_classes)

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    mean_train_losses = []
    mean_valid_losses = []
    accs = []

    C = torch.zeros((num_classes, num_classes))

    for k in range(num_classes):
        num_examples = 0
        for i, (data, labels) in enumerate(trusted_loader):
            data_k = data[labels.numpy() == k]
            num_examples += len(data_k)
            if hasattr(fu, "predict_proba"):
                if not len(data_k.numpy()) == 0:
                    C[k] += np.sum(fu.predict_proba(data_k.numpy()), axis=0)
            else:
                C[k] += torch.sum(torch.nn.functional.softmax(fu(data_k), dim=1), axis=0)
        if num_examples == 0:
            C[k] = torch.ones(num_classes) / num_classes
        else:
            C[k] = C[k] / num_examples

    C = C.detach()

    print(C)
    print(C.t())

    for epoch in range(epochs):
        model.train()

        train_losses = []
        valid_losses = []

        valid_preds = []
        valid_labels = []

        for i, (data, (labels, is_corrupteds)) in enumerate(total_loader):

            optimizer.zero_grad()

            outputs = model(data)

            loss_trusted = (cross_entropy_loss(outputs, labels) * (1 - is_corrupteds)).sum()

            loss_untrusted = (nll_loss(torch.log(torch.matmul(torch.nn.functional.softmax(outputs, dim=1), C)),
                                       labels) * is_corrupteds).sum()

            loss = (loss_trusted + loss_untrusted) / len(data)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for i, (data, labels) in enumerate(valid_loader):
                outputs = model(data)
                loss = cross_entropy_loss(outputs, labels).mean()

                valid_losses.append(loss.item())

                valid_preds.append(torch.nn.functional.softmax(outputs, dim=1).numpy()[:, 1])
                valid_labels.append(labels.numpy())

            mean_train_losses.append(np.mean(train_losses))
            mean_valid_losses.append(np.mean(valid_losses))

        acc = accuracy_score(np.concatenate(valid_labels), np.concatenate(valid_preds) > 0.5)

        accs.append(acc)

        print('c : epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}'
              .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses), acc))

    return model, pd.DataFrame(list(zip(mean_train_losses, mean_valid_losses, accs)),
                               columns=["mean_train_losses", "mean_valid_losses", "accs"]), pd.DataFrame(C)


def loop(dir, trusted, untrusted, test, optimizer, beta_batch_size, batch_size, beta_epochs, epochs,
         beta_learning_rate, learning_rate, beta_weight_decay, weight_decay, beta_hidden_size, hidden_size, calibration_method):

    print("ft-torched")
    ft_torched, ft_torched_data = normal(
        trusted, test, optimizer, beta_batch_size[0], beta_epochs[0], beta_learning_rate[0], beta_weight_decay[0], beta_hidden_size[0])
    ft_torched_data.to_csv("{}/ft-torched-perfs.csv".format(dir), index=False)

    print("fu-torched")
    fu_torched, fu_torched_data = normal(
        untrusted, test, optimizer, beta_batch_size[1], beta_epochs[1], beta_learning_rate[1], beta_weight_decay[1], beta_hidden_size[1])
    fu_torched_data.to_csv("{}/fu-torched-perfs.csv".format(dir), index=False)

    print("full-torched")
    full_torched, full_torched_data, full_torched_beta = irbl(
        trusted, untrusted, test, ft_torched, fu_torched, optimizer, batch_size, epochs, learning_rate, weight_decay, hidden_size)
    full_torched_data.to_csv("{}/full-torched-perfs.csv".format(dir), index=False)
    full_torched_beta.to_csv("{}/full-torched-beta.csv".format(dir), index=False, header=False)

    print("ft-calibrated")
    ft_torched_calibrated, ft_torched_calibrated_data = normal_sklearn(
        trusted, test, NeuralNetClassifier(
            module=LogisticRegressionSoftmax,
            module__input_size=len(trusted[0][0]),
            module__output_size=2,
            max_epochs=beta_epochs[0],
            train_split=None,
            lr=beta_learning_rate[0],
            batch_size=beta_batch_size[0],
            optimizer__weight_decay=beta_weight_decay[0],
            iterator_train__shuffle=True,
            verbose=0))
    ft_torched_calibrated_data.to_csv("{}/ft-torched-calibrated-perfs.csv".format(dir), index=False)

    print("fu-calibrated")
    fu_torched_calibrated, fu_torched_calibrated_data = normal_sklearn(
        untrusted, test, NeuralNetClassifier(
            module=LogisticRegressionSoftmax,
            module__input_size=len(trusted[0][0]),
            module__output_size=2,
            max_epochs=beta_epochs[1],
            train_split=None,
            lr=beta_learning_rate[1],
            batch_size=beta_batch_size[1],
            optimizer__weight_decay=beta_weight_decay[1],
            iterator_train__shuffle=True,
            verbose=0)
    )
    fu_torched_calibrated_data.to_csv("{}/fu-torched-calibrated-perfs.csv".format(dir), index=False)

    print("full-calibrated")
    full_torched_calibrated, full_torched_data_calibrated, full_torched_beta_calibrated = irbl(
        trusted, untrusted, test, ft_torched_calibrated, fu_torched_calibrated, optimizer, batch_size, epochs, learning_rate, weight_decay, hidden_size)
    full_torched_data_calibrated.to_csv("{}/full-torched-calibrated-perfs.csv".format(dir), index=False)
    full_torched_beta_calibrated.to_csv("{}/full-torched-calibrated-beta.csv".format(dir), index=False, header=False)

    print("glc")
    _, glc_data, C = glc(
        trusted, untrusted, test, fu_torched, optimizer, batch_size, epochs, learning_rate, weight_decay, hidden_size)
    glc_data.to_csv("{}/glc-perfs.csv".format(dir), index=False)
    C.to_csv("{}/glc-beta.csv".format(dir), index=False, header=False)

    print("mixed")
    _, mixed_data = normal(torch.utils.data.ConcatDataset([trusted, untrusted]), test, optimizer, batch_size, epochs,
                           learning_rate, weight_decay, hidden_size)
    mixed_data.to_csv("{}/mixed-perfs.csv".format(dir), index=False)

    print("symetric")
    _, symetric_data = normal(torch.utils.data.ConcatDataset([trusted, untrusted]), test, optimizer, batch_size, epochs,
                   learning_rate, weight_decay, hidden_size, loss="unhinged")
    symetric_data.to_csv("{}/symetric-perfs.csv".format(dir), index=False)

    return


def learning_curve_plot(figdir, resdir, name, p, q, criteria):

    total = pd.read_csv("{}/{}/total-perfs.csv".format(resdir, name))

    figures_directory = "{}/{}-{}-{}".format(figdir, name, p, q)
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

    results_directory = "{}/{}-{}-{}".format(resdir, name, p, q)

    ftt = pd.read_csv("{}/ft-torched-perfs.csv".format(results_directory))
    fut = pd.read_csv("{}/fu-torched-perfs.csv".format(results_directory))
    bt = pd.read_csv("{}/full-torched-perfs.csv".format(results_directory))
    btc = pd.read_csv("{}/full-torched-calibrated-perfs.csv".format(results_directory))
    mixed = pd.read_csv("{}/mixed-perfs.csv".format(results_directory))
    glc = pd.read_csv("{}/glc-perfs.csv".format(results_directory))
    symetric = pd.read_csv("{}/symetric-perfs.csv".format(results_directory))

    if criteria == "mean_valid_losse":

        ftt_error = ftt[criteria + "s"]
        fut_error = fut[criteria + "s"]
        bt_error = bt[criteria + "s"]
        btc_error = btc[criteria + "s"]
        mixed_error = mixed[criteria + "s"]
        glc_error = glc[criteria + "s"]
        symetric_error = symetric[criteria + "s"]
        total_error = total[criteria + "s"]
    
    else:

        ftt_error = 1 - ftt[criteria + "s"]
        fut_error = 1 - fut[criteria + "s"]
        bt_error = 1 - bt[criteria + "s"]
        btc_error = 1 - btc[criteria + "s"]
        mixed_error = 1 - mixed[criteria + "s"]
        glc_error = 1 - glc[criteria + "s"]
        symetric_error = 1 - symetric[criteria + "s"]
        total_error = 1 - total[criteria + "s"]

    fig, ax = plt.subplots()
    ax.set_xlabel("epochs")
    ax.set_xticks(range(len(ftt_error)))
    ax.set_xticklabels(range(1,len(ftt_error)+1))
    ax.set_ylabel("error")
    ax.plot(ftt_error, label='trusted')
    ax.plot(fut_error, label='untrtusted')
    ax.plot(bt_error, label='irbl')
    ax.plot(btc_error, label='irblc')
    ax.plot(mixed_error, label='mixed')
    ax.plot(glc_error, label='glc')
    ax.plot(symetric_error, label='symmetric')
    ax.plot(total_error, label='total')
    ax.legend()
    fig.savefig("{}/learning-curve-{}.pdf".format(figures_directory, criteria), bbox = 'tight', bbox_inches="tight", format="pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlabel("epochs")
    ax.set_xticks(range(len(ftt_error)))
    ax.set_xticklabels(range(1,len(ftt_error)+1))
    ax.set_ylabel("loss")
    ax.plot(btc_error, label='irbl', color="black")
    ax.plot(total_error, label='total', color="black", linestyle="--")
    ax.plot(mixed_error, label='mixed', color="black", linestyle="-.")
    ax.plot(ftt_error, label='trusted', color="black", linestyle=":")
    ax.plot(fut_error, label='untrusted', color="black", linestyle="--",marker=".")
    ax.legend(loc = 'upper right')
    fig.savefig("{}/learning-curve-simple-{}.pdf".format(figures_directory, criteria), bbox = 'tight', bbox_inches="tight", format="pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlabel("epochs")
    ax.set_xticks(range(len(ftt_error)))
    ax.set_xticklabels(range(1,len(ftt_error)+1))
    ax.set_ylabel("loss")
    ax.plot(btc_error, label='irbl', color="black")
    ax.plot(total_error, label='total', color="black", linestyle="--")
    ax.plot(glc_error, label='glc', color="black", linestyle="-.")
    ax.plot(symetric_error, label='rll', color="black", linestyle=":")
    ax.legend(loc = 'upper right')
    fig.savefig("{}/learning-curve-competitors-{}.pdf".format(figures_directory, criteria), bbox = 'tight', bbox_inches="tight", format="pdf")
    plt.close(fig)

def hist_plot(figdir, resdir, name, p, q):

    figures_directory = "{}/{}-{}-{}".format(figdir, name, p, q)
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

    results_directory = "{}/{}-{}-{}".format(resdir, name, p, q)

    flipped = pd.read_csv("{}/flipped.csv".format(results_directory)).to_numpy().flatten()
    bt = pd.read_csv("{}/full-torched-beta.csv".format(results_directory)).to_numpy().flatten()
    btc = pd.read_csv("{}/full-torched-calibrated-beta.csv".format(results_directory)).to_numpy().flatten()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist([bt[flipped == 0], bt[flipped == 1]], label=["cleaned", "corrupted"], bins=20, color = ["lightgray","dimgray",])
    ax.legend(loc = 'upper right')
    fig.savefig("{}/full-torched-hist.pdf".format(figures_directory), bbox = 'tight', bbox_inches="tight", format="pdf")
    plt.close(fig)

    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.hist([btc[flipped == 0], btc[flipped == 1]], #btc[flipped == 2]],
             label=["cleaned", "corrupted"], bins=20, color = ["lightgray","dimgray",])
    ax1.legend(loc = 'upper right')
    fig1.savefig("{}/full-torched-calibrated-hist.pdf".format(figures_directory), bbox = 'tight', bbox_inches="tight", format="pdf")
    plt.close(fig1)

    return


def box_plot2(figdir, resdir, name, p, qs):

    figures_directory = "{}/{}-{}".format(figdir, name, p)
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

    bt_list = []
    bt_f_list = []
    bt_t_list = []
    btc_list = []
    btc_f_list = []
    btc_t_list = []

    for q_idx, q in enumerate(qs):

        results_directory = "{}/{}-{}-{}".format(resdir, name, p, q)

        flipped = pd.read_csv("{}/flipped.csv".format(results_directory)).to_numpy().flatten()
        bt = pd.read_csv("{}/full-torched-beta.csv".format(results_directory)).to_numpy().flatten()
        btc = pd.read_csv("{}/full-torched-calibrated-beta.csv".format(results_directory)).to_numpy().flatten()

        bt_list.append(bt[flipped == 0])
        bt_f_list.append(bt[flipped == 1])
        bt_t_list.append(bt[flipped == 2])
        btc_list.append(btc[flipped == 0])
        btc_f_list.append(btc[flipped == 1])
        btc_t_list.append(btc[flipped == 2])

    c = 'lightgray'
    c0_dict = {
        'patch_artist': True,
        'boxprops': dict(facecolor=c,color="black"),
        'capprops': dict(color="black"),
        'flierprops': dict(color="black"),
        'medianprops': dict(color="black"),
        'whiskerprops': dict(color="black")}

    c = 'dimgray'
    c1_dict = {
        'patch_artist': True,
        'boxprops': dict(facecolor=c,color="black"),
        'capprops': dict(color="black"),
        'flierprops': dict(color="black"),
        'medianprops': dict(color="black"),
        'whiskerprops': dict(color="black")}

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlabel("q = 1-r")
    bp1 = ax.boxplot(bt_list[::-1], showfliers=False, labels=sorted(qs), **c0_dict)
    bp2 = ax.boxplot(bt_f_list[::-1], showfliers=False, labels=sorted(qs), **c1_dict)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['cleaned', 'corrupted'], loc='upper right')
    fig.savefig("{}/full-torched-box-plot-2.pdf".format(figures_directory), bbox = 'tight', bbox_inches="tight", format="pdf")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.set_xlabel("q = 1-r")
    bp1 = ax2.boxplot(btc_list[::-1], showfliers=False, labels=sorted(qs), **c0_dict)
    bp2 = ax2.boxplot(btc_f_list[::-1], showfliers=False, labels=sorted(qs), **c1_dict)
    ax2.legend([bp1["boxes"][0], bp2["boxes"][0]], ['cleaned', 'corrupted'], loc='upper right')
    fig2.savefig("{}/full-torched-calibrated-box-plot-2.pdf".format(figures_directory), bbox = 'tight', bbox_inches="tight", format="pdf")
    plt.close(fig2)

    return


def mean_area_under_error_curve(resdir, criteria):

    results = pd.read_csv("{}/results-{}.csv".format(resdir,criteria))

    agg = results.groupby(["p", "name"]).agg(list).reset_index()
    agg["area_under_error_curve_trusted"] = agg.apply(lambda row: np.trapz(row["trusted"]), axis=1)
    agg["area_under_error_curve_untrusted"] = agg.apply(lambda row: np.trapz(row["untrusted"]), axis=1)
    agg["area_under_error_curve_irbl"] = agg.apply(lambda row: np.trapz(row["irbl"]), axis=1)
    agg["area_under_error_curve_irblc"] = agg.apply(lambda row: np.trapz(row["irblc"]), axis=1)
    agg["area_under_error_curve_glc"] = agg.apply(lambda row: np.trapz(row["glc"]), axis=1)
    agg["area_under_error_curve_mixed"] = agg.apply(lambda row: np.trapz(row["mixed"]), axis=1)
    agg["area_under_error_curve_symetric"] = agg.apply(lambda row: np.trapz(row["symetric"]), axis=1)
    agg["area_under_error_curve_total"] = agg.apply(lambda row: np.trapz(row["total"]), axis=1)

    final = agg.groupby("p").mean().reset_index()

    final.to_csv("{}/area-{}.csv".format(resdir,criteria), index=False)

def wilcoxon_area_under_error_curve(resdir, criteria):

    results = pd.read_csv("{}/results-{}.csv".format(resdir,criteria))

    agg = results.groupby(["p", "name"]).agg(list).reset_index()
    agg["area_under_error_curve_trusted"] = agg.apply(lambda row: np.trapz(row["trusted"]), axis=1)
    agg["area_under_error_curve_untrusted"] = agg.apply(lambda row: np.trapz(row["untrusted"]), axis=1)
    agg["area_under_error_curve_irbl"] = agg.apply(lambda row: np.trapz(row["irbl"]), axis=1)
    agg["area_under_error_curve_irblc"] = agg.apply(lambda row: np.trapz(row["irblc"]), axis=1)
    agg["area_under_error_curve_glc"] = agg.apply(lambda row: np.trapz(row["glc"]), axis=1)
    agg["area_under_error_curve_mixed"] = agg.apply(lambda row: np.trapz(row["mixed"]), axis=1)
    agg["area_under_error_curve_symetric"] = agg.apply(lambda row: np.trapz(row["symetric"]), axis=1)
    agg["area_under_error_curve_total"] = agg.apply(lambda row: np.trapz(row["total"]), axis=1)

    agg = agg.drop(
        ["q", "trusted", "untrusted", "irbl", "irblc", "mixed", "glc", "symetric", "total"], axis=1)

    final = agg.groupby("p").agg(list).reset_index()

    final[["area_under_error_curve_irblc_glc_score","area_under_error_curve_irblc_glc_hypothesis"]] = pd.DataFrame(
        final.apply(lambda row: wilcoxon_test(row["area_under_error_curve_irblc"], row["area_under_error_curve_glc"]), axis=1).values.tolist())

    final[["area_under_error_curve_irblc_trusted_score","area_under_error_curve_irblc_trusted_hypothesis"]] = pd.DataFrame(
        final.apply(lambda row: wilcoxon_test(row["area_under_error_curve_irblc"], row["area_under_error_curve_trusted"]), axis=1).values.tolist())

    final[["area_under_error_curve_irblc_untrusted_score","area_under_error_curve_irblc_untrusted_hypothesis"]] = pd.DataFrame(
        final.apply(lambda row: wilcoxon_test(row["area_under_error_curve_irblc"], row["area_under_error_curve_untrusted"]), axis=1).values.tolist())

    final[["area_under_error_curve_irblc_mixed_score","area_under_error_curve_irblc_mixed_hypothesis"]] = pd.DataFrame(
        final.apply(lambda row: wilcoxon_test(row["area_under_error_curve_irblc"], row["area_under_error_curve_mixed"]), axis=1).values.tolist())

    final[["area_under_error_curve_irblc_symetric_score","area_under_error_curve_irblc_symetric_hypothesis"]] = pd.DataFrame(
        final.apply(lambda row: wilcoxon_test(row["area_under_error_curve_irblc"], row["area_under_error_curve_symetric"]), axis=1).values.tolist())

    final[["area_under_error_curve_irblc_total_score","area_under_error_curve_irblc_total_hypothesis"]] = pd.DataFrame(
        final.apply(lambda row: wilcoxon_test(row["area_under_error_curve_irblc"], row["area_under_error_curve_total"]), axis=1).values.tolist())

    final = final.drop(
        ["name", "area_under_error_curve_trusted", "area_under_error_curve_untrusted", "area_under_error_curve_irbl", 
        "area_under_error_curve_irblc", "area_under_error_curve_glc", "area_under_error_curve_mixed", "area_under_error_curve_symetric",
         "area_under_error_curve_total"], axis=1)

    final.to_csv("{}/wilcoxon-area-{}.csv".format(resdir,criteria), index=False)

def error_curve_plot(figdir, resdir, name, p, qs, criteria):

    figures_directory = "{}/{}-{}".format(figdir, name, p)
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

    res = pd.read_csv("{}/results-{}.csv".format(resdir,criteria))

    res = res[(res["name"] == name) & (res["p"]==p) & (res["q"].isin(qs))]

    fig, ax = plt.subplots()
    ax.set_xlabel("q = 1-r")
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels(qs)
    ax.set_ylabel("error")
    ax.plot(res["trusted"], label='trusted')
    ax.plot(res["untrusted"], label='untrusted')
    ax.plot(res["irbl"], label='irbl')
    ax.plot(res["irblc"], label='irblc')
    ax.plot(res["mixed"], label='mixed')
    ax.plot(res["glc"], label='glc')
    ax.plot(res["symetric"], label='symmetric')
    ax.plot(res["total"], label='total')
    ax.legend()
    fig.savefig("{}/error-curve-{}.pdf".format(figures_directory, criteria), bbox = 'tight', bbox_inches="tight", format="pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlabel("q = 1-r")
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels(sorted(qs))
    ax.set_ylabel("error")
    ax.plot(res["irblc"].values[::-1], label='irbl', color="black")
    ax.plot(res["total"].values[::-1], label='total', color="black", linestyle="--")
    ax.plot(res["mixed"].values[::-1], label='mixed', color="black", linestyle="-.")
    ax.plot(res["trusted"].values[::-1], label='trusted', color="black", linestyle=":")
    ax.plot(res["untrusted"].values[::-1], label='untrusted', color="black", linestyle="--",marker=".")
    ax.legend(loc = 'upper right')
    fig.savefig("{}/error-curve-simple-{}.pdf".format(figures_directory, criteria), bbox = 'tight', bbox_inches="tight", format="pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlabel("q = 1-r")
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels(sorted(qs))
    ax.set_ylabel("error")
    ax.plot(res["irblc"].values[::-1], label='irbl', color="black")
    ax.plot(res["total"].values[::-1], label='total', color="black", linestyle="--")
    ax.plot(res["glc"].values[::-1], label='glc', color="black", linestyle="-.")
    ax.plot(res["symetric"].values[::-1], label='rll', color="black", linestyle=":")
    ax.legend(loc = 'upper right')
    fig.savefig("{}/error-curve-competitors-{}.pdf".format(figures_directory, criteria), bbox = 'tight', bbox_inches="tight", format="pdf")
    plt.close(fig)


def generate_results(resdir, names, ps, qs, criteria):

    ftt_error_list = []
    fut_error_list = []
    bt_error_list = []

    btc_error_list = []

    mixed_error_list = []

    glc_error_list = []

    symetric_error_list = []

    total_error_list = []

    name_list = []
    p_list = []
    q_list = []

    for _, name in enumerate(names):

        for _, p in enumerate(ps):

            for _, q in enumerate(qs):

                complete_resdir = "{}/{}-{}-{}".format(resdir, name, p, q)

                bt = pd.read_csv("{}/full-torched-perfs.csv".format(complete_resdir))
                ftt = pd.read_csv("{}/ft-torched-perfs.csv".format(complete_resdir))
                fut = pd.read_csv("{}/fu-torched-perfs.csv".format(complete_resdir))

                btc = pd.read_csv("{}/full-torched-calibrated-perfs.csv".format(complete_resdir))

                mixed = pd.read_csv("{}/mixed-perfs.csv".format(complete_resdir))

                glc = pd.read_csv("{}/glc-perfs.csv".format(complete_resdir))

                symetric = pd.read_csv("{}/symetric-perfs.csv".format(complete_resdir))

                total = pd.read_csv("{}/{}/total-perfs.csv".format(resdir, name))

                if criteria == "mean_valid_losse":

                    ftt_error = np.min(ftt[criteria + "s"])
                    fut_error = np.min(fut[criteria + "s"])
                    bt_error = np.min(bt[criteria + "s"])
                    btc_error = np.min(btc[criteria + "s"])
                    mixed_error = np.min(mixed[criteria + "s"])
                    glc_error = np.min(glc[criteria + "s"])
                    symetric_error = np.min(symetric[criteria + "s"])
                    total_error = np.min(total[criteria + "s"])
                
                else:

                    ftt_error = np.min(1 - ftt[criteria + "s"])
                    fut_error = np.min(1 - fut[criteria + "s"])
                    bt_error = np.min(1 - bt[criteria + "s"])
                    btc_error = np.min(1 - btc[criteria + "s"])
                    mixed_error = np.min(1 - mixed[criteria + "s"])
                    glc_error = np.min(1 - glc[criteria + "s"])
                    symetric_error = np.min(1 - symetric[criteria + "s"])
                    total_error = np.min(1 - total[criteria + "s"])

                ftt_error_list.append(ftt_error)
                fut_error_list.append(fut_error)
                bt_error_list.append(bt_error)

                btc_error_list.append(btc_error)

                mixed_error_list.append(mixed_error)

                glc_error_list.append(glc_error)

                symetric_error_list.append(symetric_error)

                total_error_list.append(total_error)

                name_list.append(name)
                p_list.append(p)
                q_list.append(q)

    res = pd.DataFrame(list(zip(name_list, p_list, q_list, ftt_error_list, fut_error_list, bt_error_list, btc_error_list, mixed_error_list, glc_error_list, symetric_error_list, total_error_list)),
                       columns=["name", "p", "q", "trusted", "untrusted", "irbl", "irblc", "mixed", "glc", "symetric", "total"])

    res.to_csv("{}/results-{}.csv".format(resdir,criteria), index=False)


def generate_wilcoxon(resdir,criteria):

    results = pd.read_csv("{}/results-{}.csv".format(resdir,criteria))

    agg = results.groupby(["p", "q"]).agg(list).reset_index()

    agg[["irblc_glc_score", "irblc_glc_hypothesis"]] = pd.DataFrame(
        agg.apply(lambda row: wilcoxon_test(row["irblc"], row["glc"]), axis=1).values.tolist())

    agg[["irblc_mixed_score", "irblc_mixed_hypothesis"]] = pd.DataFrame(
        agg.apply(lambda row: wilcoxon_test(row["irblc"], row["mixed"]), axis=1).values.tolist())

    agg[["irblc_trusted_score", "irblc_trusted_hypothesis"]] = pd.DataFrame(
        agg.apply(lambda row: wilcoxon_test(row["irblc"], row["trusted"]), axis=1).values.tolist())

    agg[["irblc_untrusted_score", "irblc_untrusted_hypothesis"]] = pd.DataFrame(
        agg.apply(lambda row: wilcoxon_test(row["irblc"], row["untrusted"]), axis=1).values.tolist())

    agg[["irblc_symetric_score", "irblc_symetric_hypothesis"]] = pd.DataFrame(
        agg.apply(lambda row: wilcoxon_test(row["irblc"], row["symetric"]), axis=1).values.tolist())

    agg[["irblc_total_score", "irblc_total_hypothesis"]] = pd.DataFrame(
        agg.apply(lambda row: wilcoxon_test(row["irblc"], row["total"]), axis=1).values.tolist())

    agg[["mixed_total_score", "mixed_total_hypothesis"]] = pd.DataFrame(
        agg.apply(lambda row: wilcoxon_test(row["mixed"], row["total"]), axis=1).values.tolist())

    agg = agg.reindex(agg.irblc_glc_score.abs().sort_values(ascending=False).index).drop(
        ["name", "trusted", "untrusted", "irbl", "irblc", "mixed", "glc", "symetric", "total"], axis=1)

    agg.to_csv("{}/wilcoxon-{}.csv".format(resdir,criteria), index=False)

    return ((agg["irblc_glc_score"]>1.96).sum(),(agg["irblc_glc_score"]<-1.96).sum())


def wilcoxon_plot(figdir, resdir, criteria, comp1, comp2):

    agg = pd.read_csv("{}/wilcoxon-{}.csv".format(resdir,criteria))

    ps_dict= {0.02:0,0.05:0.5,0.1:1,0.25:1.5}
    ps = agg["p"].sort_values().unique()
    qs = agg["q"].sort_values().unique()
    score_col = "{}_{}_score".format(comp1,comp2)
    scores = agg
    fig, ax = plt.subplots(figsize=(5,2.5))

    ties = scores[(scores[score_col]<1.96)&(scores[score_col]>-1.96)]
    losses= scores[scores[score_col]<-1.96]
    wins = scores[scores[score_col]>1.96]

    ax.scatter(wins["q"],np.array([ps_dict[x] for x in wins["p"].values]),color="black",facecolor="white",label="win")
    ax.scatter(ties["q"],np.array([ps_dict[x] for x in ties["p"].values]),color="black",marker=".",s=1,label="tie")
    ax.scatter(losses["q"],np.array([ps_dict[x] for x in losses["p"].values]),color="black",label="loss")

    ax.set_xlabel("q = 1-r")

    ax.set_ylabel("p")

    ax.set_xticks(qs)
    ax.set_yticks(np.array([ps_dict[x] for x in ps]))

    ax.set_xticklabels(qs)
    ax.set_yticklabels(ps)

    plt.tight_layout()

    filename = "{}/wilcoxon-{}-{}-{}.pdf".format(figdir, criteria, comp1, comp2)
    fig.savefig(filename, bbox='tight', bbox_inches="tight", format="pdf")
    plt.close(fig)


optimizer = "sgd"
beta_batch_size = (24, 24)
batch_size = 24
beta_epochs = (20, 20)
epochs = 20
beta_learning_rate = (0.005, 0.005)
learning_rate = 0.005
beta_weight_decay = (1e-6, 1e-6)
weight_decay = 1e-6
beta_hidden_size = (0, 0)
hidden_size = 0
calibration_method = "isotonic"

dss = [
    ad,
    banknote,
    ibn_sina,
    eeg,
    ijcnn1,
    adult,
    phishing,
    spam,
    musk,
    australian,
    diabetes,
    breast,
    german,
    fourclass,
    svmguide3,
    svmguide1,
    web,
    hiva,
    sylva,
    zebra,
]

names = [
    "ad",
    "banknote",
    "ibn_sina",
    "eeg",
    "ijcnn1",
    "adult",
    "phishing",
    "spam",
    "musk",
    "australian",
    "diabetes",
    "breast",
    "german",
    "fourclass",
    "svmguide3",
    "svmguide1",
    "web",
    "hiva",
    "sylva",
    "zebra",
]

cr_kinds = [noisy_completly_at_random, noisy_not_at_random]
cr_names = ["ncar","nnar"]


ps = [0.02,0.05,0.1,0.25]
qs = [1.0, 0.9, 0.8, 0.7,0.6, 0.5, 0.4,0.3,0.2,0.1, 0.0]

for cr_idx, cr_kind in enumerate(cr_kinds):

    base_dir = cr_names[cr_idx]

    for ds_idx, ds_lazy in enumerate(dss):

        name = names[ds_idx]

        print(name)

        ds_dir = "{}/{}".format(base_dir, name)
        if not os.path.exists(ds_dir):
            os.makedirs(ds_dir)

        dataset = ds_lazy()
        train, test = split_scale_dataset(dataset, 0.2)

        print("total")
        total_model, total = normal(train, test, optimizer, batch_size, epochs, learning_rate, weight_decay, hidden_size)
        total.to_csv("{}/total-perfs.csv".format(ds_dir), index=False)

        for _, p in enumerate(ps):

            trusted, untrusted = split_dataset(train, (1 - p))

            for _, q in enumerate(qs):

                print(name, p, q)

                dir = "{}-{}-{}".format(ds_dir, p, q)
                if not os.path.exists(dir):
                    os.makedirs(dir)

                corrupted = corrupt_dataset(untrusted, cr_kind, 1 - q)

                # Use with NNAR
                # corrupted = corrupt_dataset(untrusted, lambda y,ratio: cr_kind(torch.nn.functional.softmax(total_model(torch.from_numpy(untrusted[:][0])),dim=1)[:,1].detach().numpy(),y,ratio), 1 - q)

                print(np.sum(corrupted[:][1] != untrusted[:][1]) / len(corrupted[:][1] != untrusted[:][1]))

                pd.Series(np.full(len(trusted), 2.0)).append(pd.Series(corrupted[:][1] != untrusted[:][1]).astype(int)).to_csv(
                    "{}/flipped.csv".format(dir), index=False, header=False)

                loop(dir, trusted, corrupted, test, optimizer, beta_batch_size,
                     batch_size, beta_epochs, epochs, beta_learning_rate,
                     learning_rate, beta_weight_decay, weight_decay, beta_hidden_size, hidden_size, calibration_method)

                hist_plot("{}-figures".format(base_dir), base_dir, name, p, q)

                learning_curve_plot("{}-figures".format(base_dir), base_dir, name, p, q, "mean_valid_losse")
                learning_curve_plot("{}-figures".format(base_dir), base_dir, name, p, q, "acc")

            generate_results(base_dir, [name], [p], qs, "acc")
            error_curve_plot("{}-figures".format(base_dir), base_dir, name, p, qs, "acc")

            box_plot2("{}-figures".format(base_dir), base_dir, name, p, qs)

    generate_results(base_dir, names, ps, qs, "acc")
    generate_wilcoxon(base_dir, "acc")
    wilcoxon_plot("{}-figures".format(base_dir),base_dir,"acc","irblc","glc")
    wilcoxon_plot("{}-figures".format(base_dir),base_dir,"acc","irblc","mixed")
    wilcoxon_plot("{}-figures".format(base_dir),base_dir,"acc","irblc","trusted")
    wilcoxon_plot("{}-figures".format(base_dir),base_dir,"acc","irblc","untrusted")
    wilcoxon_plot("{}-figures".format(base_dir),base_dir,"acc","irblc","total")
    wilcoxon_plot("{}-figures".format(base_dir),base_dir,"acc","irblc","symetric")

    results = pd.read_csv("{}/results-{}.csv".format(base_dir,"acc"))

    method_names = ["trusted","rll","irbl","glc","mixed","total"]
    final = results.groupby(["name"]).sum().reset_index()
    avranks = friedman_test(final["trusted"].values,final["symetric"].values,final["irblc"].values,final["glc"].values,
    final["mixed"].values,final["total"].values,reverse=False)[2]

    cd = Orange.evaluation.compute_CD(avranks, 20)
    Orange.evaluation.graph_ranks(avranks, method_names, cd=cd, width=6, textspace=1)
    plt.savefig("{}/cd.pdf".format("{}-figures".format(base_dir), "acc"), bbox = 'tight', bbox_inches="tight", format="pdf")

    print(wilcoxon_test(final["irblc"].values,final["glc"].values))
    print(wilcoxon_test(final["irblc"].values,final["mixed"].values))

    generate_results(base_dir, names, ps, qs, "acc")
    results = pd.read_csv("{}/results-{}.csv".format(base_dir,"acc"))
    results["irblc"] = 100*(1 - results["irblc"])
    results["trusted"] = 100*(1 - results["trusted"])
    results["symetric"] = 100*(1 - results["symetric"])
    results["glc"] = 100*(1 - results["glc"])
    results["mixed"] = 100*(1 - results["mixed"])
    results["total"] = 100*(1 - results["total"])
    results = results.drop(["untrusted","irbl"],axis=1)
    results.groupby(["p","name"]).agg(["mean","std"]).drop("q",axis=1,level=0).reset_index().groupby("p").mean().to_csv("{}/aggregated-results-{}.csv".format(base_dir,"acc"), index=False)
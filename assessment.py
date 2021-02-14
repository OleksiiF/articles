#!/usr/bin/env python3
import sys
import re
import os
import string
from collections import Counter

import spacy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as functi
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


tok = spacy.load("en_core_web_sm")
stop_words = tok.Defaults.stop_words


class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(
            self.X[idx][0].astype(np.int32)
        ), self.y[idx], self.X[idx][1]


def tokenize(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile(
        '[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]'
    )  # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())

    return [
        token.text
        for token in tok.tokenizer(nopunct)
        if token.text not in stop_words
    ]


def get_data(root):
    raw_data = []
    subs = os.listdir(root)

    for sub in subs:
        path_files = os.listdir(f"{root}/{sub}")

        for path_file in path_files:
            with open(f"{root}/{sub}/{path_file}") as fh:
                raw_data.append((sub, fh.read()))

    return raw_data


def encode_sentence(text, vocab2index, N):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array(
        [vocab2index.get(word, vocab2index["UNK"])
        for word in tokenized]
    )
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]

    return encoded, length


def train_model(model, train_dl, val_dl, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.Adam(parameters, lr=lr)
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.01)

    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0

        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = functi.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]

        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)

        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (
            sum_loss / total, val_loss, val_acc, val_rmse))


def validation_metrics(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0

    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = functi.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        sum_rmse += np.sqrt(
            mean_squared_error(pred, y.unsqueeze(-1))
        ) * y.shape[0]

    return sum_loss / total, correct / total, sum_rmse / total


def main(root):
    # create dataframe
    raw_data = get_data(root)
    df_articles = pd.DataFrame(
        data=pd.DataFrame(
            raw_data,
            columns=['topic', 'article']
        )
    )

    df_articles['article_length'] = df_articles['article'].apply(lambda x: len(x.split()))

    # count number of occurences of each word
    counts = Counter()
    for index, row in df_articles.iterrows():
        counts.update(tokenize(row['article']))

    # deleting infrequent words
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]

    # creating vocabulary
    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    topic_vocab = {
        el: index
        for index, el in enumerate(df_articles.topic.unique())
    }

    df_articles['article_encoded'] = df_articles['article'].apply(
        lambda x: np.array(
            encode_sentence(
                x,
                vocab2index,
                np.max(df_articles['article_length'])
            )
        )
    )
    df_articles['topic_encoded'] = df_articles['topic'].map(lambda x: topic_vocab[x])

    X = list(df_articles['article_encoded'])
    y = list(df_articles['topic_encoded'])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.01)

    train_ds = ReviewsDataset(X_train, y_train)
    valid_ds = ReviewsDataset(X_valid, y_valid)

    batch_size = 1000
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)


    class LSTM_fixed_len(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.linear = nn.Linear(hidden_dim, 5)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x, l):
            x = self.embeddings(x)
            x = self.dropout(x)
            lstm_out, (ht, ct) = self.lstm(x)
            return self.linear(ht[-1])

    model_fixed = LSTM_fixed_len(len(words), 50, 50)

    train_model(model_fixed, train_dl, val_dl, lr=0.01)


if __name__ == '__main__':
    main(sys.argv[1])

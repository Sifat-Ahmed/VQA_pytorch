from collections import defaultdict

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from config import Config
from models.nlp_models import LSTM
from helper.vocab import Vocabulary
from helper.preprocessing import process_answer, load_glove_model, load_bangla_word2vec, process_sentence
from utils.utils import MetricMonitor, calculate_accuracy

from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, text, label):
        self._text = text
        self._label = label
        assert len(self._text) == len(self._label), "length of dataset is not same"

    def __len__(self):
        return len(self._text)

    def __getitem__(self, index):
        text = self._text[index]
        label = self._label[index]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

def parse_dataset(csv_path):
    df = pd.read_csv(csv_path, sep=',')
    text, label = list(), list()
    pos, neg = 0, 0


    for i in range(len(df)):
        txt = df.loc[i, 'Review']
        lbl = df.loc[i, 'Rating']
        words = [word for word in word_tokenize(txt) if word.isalpha()]
        if len(words) > 1000:
            continue
        txt = " ".join(words)
        lbl = 1 if lbl > 3 else 0

        text.append(txt)
        label.append(lbl)


    return text, label

def train(cfg,
          epoch,
          train_dloader,
          model,
          criterion,
          optimizer
        ):
    model.train()

    ## Defining the print stream
    ## This will hold all the metrices
    metric_monitor = MetricMonitor()
    # loader = self.train_loader
    # start_epoch = self.epoch

    stream = tqdm(train_dloader, position=0, leave=True, colour='green')


    for idx, (text, label) in enumerate(stream, start=1):
        text = text.to(cfg.device, non_blocking=cfg.non_blocking)
        label = label.to(cfg.device, non_blocking=cfg.non_blocking).float().view(-1, 1)

        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(output, label)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", acc)

        stream.set_description(
            "Epoch: {epoch}. Train. {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor)
        )

    return metric_monitor

@torch.no_grad()
def valid(cfg,
          epoch,
          valid_dloader,
          model,
          criterion,

          ):
    model.eval()

    ## Defining the print stream
    ## This will hold all the metrices
    metric_monitor = MetricMonitor()
    # loader = self.train_loader
    # start_epoch = self.epoch

    stream = tqdm(valid_dloader, position=0, leave=True, colour='red')

    for idx, (text, label) in enumerate(stream, start=1):
        text = text.to(cfg.device, non_blocking=cfg.non_blocking)
        label = label.to(cfg.device, non_blocking=cfg.non_blocking).float().view(-1, 1)

        output = model(text)
        loss = criterion(output, label)

        acc = calculate_accuracy(output, label)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", acc)

        stream.set_description(
            "Epoch: {epoch}. Valid. {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor)
        )

    return metric_monitor


if __name__  == "__main__":

    text, label = parse_dataset(r'/home/workstaion/workspace/potatochips/vqa/Python/reviews.csv')
    #print(text[0:5])
    #print(label[:5])

    cfg = Config()

    vocab = Vocabulary(maxlen=1000)
    text_seq = vocab.get_sequences(text)
    vocab_size = vocab.vocab_size()

    x_train, x_test, y_train, y_test = train_test_split(text_seq, label,
                                                        test_size=cfg.validation_size,
                                                        stratify=label)

    train_dataset = TextDataset(x_train, y_train)
    val_dataset = TextDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, shuffle=cfg.shuffle, batch_size=cfg.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=cfg.shuffle, batch_size=cfg.batch_size)


    model = LSTM(out_features=1, vocab_size=vocab_size).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCEWithLogitsLoss().to(cfg.device)

    history = defaultdict()
    history['val_loss'], history['train_loss'] = list(), list()
    history['val_acc'], history['train_acc'] = list(), list()

    BEST_LOSS = np.inf


    for epoch in range(0, 20):
        train_hist = train(cfg, epoch, train_loader, model, criterion, optimizer)
        history['train_loss'].append(train_hist.get('Loss'))
        history['train_acc'].append(train_hist.get('Accuracy'))

        test_hist = valid(cfg, epoch, val_loader, model, criterion)

        if test_hist.get('Loss') < BEST_LOSS:
            print('Saving model...')
            BEST_LOSS = test_hist.get('Loss')
        history['val_loss'].append(test_hist.get('Loss'))
        history['val_acc'].append(test_hist.get('Accuracy'))


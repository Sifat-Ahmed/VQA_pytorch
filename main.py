from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import os, gc
from collections import defaultdict
import numpy as np
from config import Config
from helper.json_parser import ParseJson
from Datasets.dataset import LoadDataset
from sklearn.preprocessing import LabelEncoder
from helper.preprocessing import load_glove_model, load_bangla_word2vec
from helper.vocab import Vocabulary
from sklearn.model_selection import train_test_split
from models.nlp_models import LSTM
from models.vision_models import ResNet50
from models.attention_models import AttentionNet, HierAttnNet
from utils.utils import MetricMonitor, calculate_accuracy

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
cudnn.benchmark = True



def train(cfg,
          epoch,
          train_dloader,
          vision_model,
          nlp_model,
          attention_model,
          criterion,
          optimizer
        ):
    vision_model.train()
    nlp_model.train()
    attention_model.train()

    ## Defining the print stream
    ## This will hold all the metrices
    metric_monitor = MetricMonitor()
    # loader = self.train_loader
    # start_epoch = self.epoch

    stream = tqdm(train_dloader, position=0, leave=True, colour='green')
    # print("Start/Continue training from epoch {}".format(start_epoch))
    # print("<------------------------Start Training------------------------------------>")

    # for epoch in tqdm(range(start_epoch, self.num_epochs), leave = True, position = 0, colour = 'green'):

    running_loss, running_acc, num_updates, run_acc, batch = 0.0, 0.0, 0.0, 0.0, 0

    for idx, (text, image, ans) in enumerate(stream, start=1):
        text = text.to(cfg.device, non_blocking=cfg.non_blocking)
        image = image.to(cfg.device, non_blocking=cfg.non_blocking)
        ans = ans.to(cfg.device, non_blocking=cfg.non_blocking).float().view(-1, 1)

        optimizer.zero_grad()

        image_embed = vision_model(image)
        question_embed = nlp_model(text)
        output = attention_model(image_embed, question_embed)
        #_, y_pred = torch.max(output, 1)

        loss = criterion(output, ans)

        # backprop
        loss.backward()
        optimizer.step()

        # correct = (y_pred == ans).float()
        # run_acc = (correct.sum() / len(correct)).item()
        acc = calculate_accuracy(output, ans, cfg.classification_threshold)


        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", acc)

        stream.set_description(
            "Epoch: {epoch}. Train. {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor)
        )

    return metric_monitor

@torch.no_grad()
def validation(cfg,
          epoch,
          validation_dloader,
          vision_model,
          nlp_model,
          attention_model,
          criterion):
    vision_model.eval()
    nlp_model.eval()
    attention_model.eval()

    metric_monitor = MetricMonitor()
    stream = tqdm(validation_dloader, position=0, leave=True, colour='red')

    for idx, (text, image, ans) in enumerate(stream, start=1):
        text = text.to(cfg.device, non_blocking = cfg.non_blocking)
        image = image.to(cfg.device, non_blocking = cfg.non_blocking)
        ans = ans.to(cfg.device, non_blocking = cfg.non_blocking).float().view(-1, 1)


        image_embed = vision_model(image)
        question_embed = nlp_model(text)
        output = attention_model(image_embed, question_embed)
        # _, y_pred = torch.max(output, 1)

        loss = criterion(output, ans)
        acc = calculate_accuracy(output, ans, cfg.classification_threshold)
        # correct=(y_pred==ans).float()
        # acc = (correct.sum()/len(correct)).item()

        #print(output)

        metric_monitor.update("Loss", loss)
        metric_monitor.update("Accuracy", acc)
        stream.set_description(
            "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )

    return metric_monitor


def main():
    ## creating a cofig object for future usage
    ## we will keep sending this reference to other class instances
    cfg = Config()
    ## creating a json parser object
    json_parser = ParseJson(cfg)
    ## getting the question, image and the answer here
    question, image, answer = json_parser.get_ques_im_ans()

    ## OneHotEncoding of the labels. In our case we have two labels. So "yes" is [0, 1] and "No" is [1, 0]
    one_hot_enc = LabelEncoder()
    answer = one_hot_enc.fit_transform(np.array(answer))


    ## Build a vocabulary and get the sequences
    #glove_bengali = load_glove_model(cfg.glove_path)
    ## Creating a vocabulary object reference
    vocab_builder = Vocabulary(maxlen=cfg.max_len)
    ## building the vocabulary and transforming into sequence
    sequences = vocab_builder.get_sequences(question)

    vocab_size = vocab_builder.vocab_size()

    ## creating a training and validation set
    ## size is defined in the config
    train_ques, val_ques, train_img, val_img, train_ans, val_ans = train_test_split(sequences, image, answer,
                                                                                   test_size=cfg.validation_size,
                                                                                    stratify=answer,
                                                                                    shuffle=True)

    ## from the remaining training data, creating a test set
    # size is defined in the config file
    # train_ques, test_ques, train_img, test_img, train_ans, test_ans = train_test_split(sequences, image, answer,
    #                                                                                test_size=cfg.test_size, stratify=answer)


    #creating dataset object
    training_dataset = LoadDataset(cfg, train_ques, train_img, train_ans, mode = "train", transform=cfg.train_transform)
    validation_dataset = LoadDataset(cfg, val_ques, val_img, val_ans, mode = "train", transform=cfg.transform)
    # test_dataset = LoadDataset(cfg, test_ques, test_img, test_ans, mode = "train")


    ## creating data Loaders
    training_dloader = DataLoader(training_dataset,
                                  batch_size=cfg.train_batch_size,
                                  shuffle = cfg.shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory = cfg.pin_memory)

    validation_dloader = DataLoader(validation_dataset,
                                  batch_size=cfg.val_batch_size,
                                  shuffle=cfg.shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=cfg.pin_memory)



    vision_model = ResNet50(out_features=cfg.vector_size).to(cfg.device)
    nlp_model = HierAttnNet(vocab_size=vocab_size, maxlen_sent=20, maxlen_doc=1, sent_hidden_dim=100, word_hidden_dim=100,
                        embed_dim=cfg.max_len, num_class=cfg.vector_size).to(cfg.device)
    attention_model = AttentionNet(input_features=cfg.vector_size).to(cfg.device)

    optimizer_parameter_group = [{'params': nlp_model.parameters()},
                                  {'params': vision_model.parameters()},
                                  {'params': attention_model.parameters()}]

    # loss function and optimizer
    criterion = nn.BCEWithLogitsLoss().to(cfg.device)  ### Loss

    optimizer = torch.optim.Adam(optimizer_parameter_group,
                                      lr=cfg.learning_rate)  ### Optimizer

    history = defaultdict()
    history['val_loss'], history['train_loss'] = list(), list()
    history['val_acc'], history['train_acc'] = list(), list()

    BEST_LOSS = np.inf

    for epoch in range(1, cfg.epochs+1):
        train_hist = train(cfg, epoch, training_dloader, vision_model, nlp_model, attention_model, criterion, optimizer)
        history['train_loss'].append(train_hist.get('Loss'))
        history['train_acc'].append(train_hist.get('Accuracy'))

        test_hist = validation(cfg, epoch, validation_dloader, vision_model, nlp_model, attention_model, criterion)

        if test_hist.get('Loss') < BEST_LOSS:
            print('Saving model...')
            BEST_LOSS = test_hist.get('Loss')
        history['val_loss'].append(test_hist.get('Loss'))
        history['val_acc'].append(test_hist.get('Accuracy'))

        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()


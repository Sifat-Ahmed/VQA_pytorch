from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import os
import numpy as np
from config import Config
from helper.json_parser import ParseJson
from Datasets.dataset import LoadDataset
from sklearn.preprocessing import OneHotEncoder
from helper.preprocessing import load_glove_model, load_bangla_word2vec
from helper.vocab import Vocabulary
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


def main():
    ## creating a cofig object for future usage
    ## we will keep sending this reference to other class instances
    cfg = Config()
    ## creating a json parser object
    json_parser = ParseJson(cfg)
    ## getting the question, image and the answer here
    question, image, answer = json_parser.get_ques_im_ans()

    ## OneHotEncoding of the labels. In our case we have two labels. So "yes" is [0, 1] and "No" is [1, 0]
    one_hot_enc = OneHotEncoder()
    answer = one_hot_enc.fit_transform(np.array(answer).reshape(-1, 1)).toarray()

    ## Build a vocabulary and get the sequences
    glove_bengali = load_glove_model(cfg.glove_path)
    ## Creating a vocabulary object reference
    vocab_builder = Vocabulary()
    ## building the vocabulary and transforming into sequence
    sequences = vocab_builder.get_sequences(question)


    ## creating a training and validation set
    ## size is defined in the config
    train_ques, val_ques, train_img, val_img, train_ans, val_ans = train_test_split(sequences, image, answer,
                                                                                   test_size=cfg.validation_size, stratify=answer)

    ## from the remaining training data, creating a test set
    # size is defined in the config file
    train_ques, test_ques, train_img, test_img, train_ans, test_ans = train_test_split(sequences, image, answer,
                                                                                   test_size=cfg.test_size, stratify=answer)


    #creating dataset object
    training_dataset = LoadDataset(cfg, train_ques, train_img, train_ans, mode = "train")
    validation_dataset = LoadDataset(cfg, val_ques, val_img, val_ans, mode = "train")
    test_dataset = LoadDataset(cfg, test_ques, test_img, test_ans, mode = "train")


    ## creating data Loaders
    training_dloader = DataLoader(training_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle = cfg.shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory = cfg.pin_memory)

    validation_dloader = DataLoader(validation_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=cfg.shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=cfg.pin_memory)



    for ques, img, ans in training_dloader:
        print(ques)
        print(ans)
        break




if __name__ == "__main__":
    main()


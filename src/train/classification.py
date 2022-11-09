import pandas as pd
import ast

from sklearn.model_selection import train_test_split

import torch


def classification_train(data_path, config):
    df = pd.read_csv(data_path)

    exp_name = config.name + "_{}".format(config.train.alpha)
    epoch = config.train.epoch
    debug = config.debug
    gpu_num = torch.cuda.device_count()
    batch_size = config.train.batch_size
    df = pd.read_csv(data_path)
    df["text"] = [ast.literal_eval(d) for d in df["text"]]
    train, validate = train_test_split(df)

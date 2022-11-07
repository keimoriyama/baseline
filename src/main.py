import pandas as pd
import ast

from dataset import SimulateDataset
from model import BaselineModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf


def main():
    data_path = "./data/train.csv"

    config = OmegaConf.load("./config/baseline.yml")

    seed_everything(config.seed)
    exp_name = config.name + "_{}".format(config.train.alpha)
    epoch = config.train.epoch
    debug = config.debug
    gpu_num = torch.cuda.device_count()
    batch_size= config.train.batch_size
    df = pd.read_csv(data_path)
    df["text"] = [ast.literal_eval(d) for d in df["text"]]
    train, validate = train_test_split(df)
    if debug:
        train = train[: 8 * 2 ]
        validate = validate[: 8 * 2]
    # データセットの用意
    train = train.reset_index()
    validate = validate.reset_index()
    train_dataset = SimulateDataset(train)
    validate_dataset = SimulateDataset(validate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers)

    # loggerの用意
    wandb_logger = WandbLogger(name=exp_name, project="baseline")
    wandb_logger.log_hyperparams(config.train)
    # 学習部分
    trainer = pl.Trainer(max_epochs=epoch, logger=wandb_logger, strategy="ddp",  gpus = gpu_num)
    model = BaselineModel(alpha=config.train.alpha,
    load_bert=True, out_dim = config.train.out_dim,
    learning_rate = config.train.learning_rate,
    hidden_dim = config.train.hidden_dim,
    dropout_rate=config.train.dropout_rate
    )
    trainer.fit(
        model,
        train_dataloader,
        validate_dataloader
    )


if __name__ == "__main__":
    main()

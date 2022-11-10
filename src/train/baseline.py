from models import FlattenModel, RandomModel

from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd
import ast

from dataset import SimulateDataset

from trainer import BaselineModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader


def baseline_train(data_path, config):
    exp_name = config.name + "_{}".format(config.train.alpha)
    epoch = config.train.epoch
    debug = config.debug
    gpu_num = torch.cuda.device_count()
    batch_size = config.train.batch_size
    df = pd.read_csv(data_path)
    df["text"] = [ast.literal_eval(d) for d in df["text"]]
    train, validate = train_test_split(df, test_size=0.2)
    validate, test = train_test_split(df, test_size=0.5)
    if debug:
        train = train[: 8 * 2]
        validate = validate[: 8 * 2]
    # データセットの用意
    train = train.reset_index()
    validate = validate.reset_index()
    train_dataset = SimulateDataset(train)
    validate_dataset = SimulateDataset(validate)
    test_dataset = SimulateDataset(test)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers
    )
    validate_dataloader = DataLoader(
        validate_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=config.dataset.num_workers
    )

    # loggerの用意
    wandb_logger = WandbLogger(name=exp_name, project="baseline")
    wandb_logger.log_hyperparams(config.train)
    # 学習部分
    checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="validation_loss",
    mode="min",
    dirpath="./model/baseline/",
    filename="model_alpha_{}_seed_{}".format(config.train.alpha, config.seed),
    save_weights_only=True
)

    trainer = pl.Trainer(
        max_epochs=epoch, logger=wandb_logger, strategy="ddp", gpus=gpu_num,callbacks=[checkpoint_callback]
    )
    if config.model == "random":
        model = RandomModel(out_dim=config.train.out_dim)
        trainer.test(model, test_dataloader)
    elif config.model =="flatten":
        model = FlattenModel(
            token_len=512,
            out_dim=config.train.out_dim,
            hidden_dim=config.train.hidden_dim,
            dropout_rate=config.train.dropout_rate,
            load_bert=True,
        )

        model = BaselineModel(alpha=config.train.alpha, model=model)
        trainer.fit(model, train_dataloader, validate_dataloader)
        trainer.test(model, test_dataloader)

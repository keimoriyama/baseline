import pandas as pd
import ast

from models import FlattenModel

from sklearn.model_selection import train_test_split
from dataset import ClassificationDataset
from trainer import ClassificationTrainer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torch.utils.data import DataLoader

def classification_train(data_path, config):
    df = pd.read_csv(data_path)

    exp_name = config.name + "_{}".format(config.train.alpha)
    epoch = config.train.epoch
    debug = config.debug
    gpu_num = torch.cuda.device_count()
    batch_size = config.train.batch_size
    out_size = max(df['attribute_id'])+1
    df = pd.read_csv(data_path)
    df["text"] = [ast.literal_eval(d) for d in df["text"]]
    train, validate = train_test_split(df, test_size=0.2, stratify = df['attribute_id'])
    validate, test = train_test_split(validate, test_size=0.5, stratify = validate['attribute_id'])
    print(len(train), len(validate), len(test))

    train_dataset = ClassificationDataset(train)
    validate_dataset = ClassificationDataset(validate)
    test_dataset = ClassificationDataset(test)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers
    )
    validate_dataloader = DataLoader(
        validate_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers
    )
    test_dataloader = DataLoader(
        validate_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers
    )
    # loggerの用意
    wandb_logger = WandbLogger(name=exp_name, project="classification")
    wandb_logger.log_hyperparams(config.train)

    checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="validation_loss",
    mode="min",
    dirpath="./model/baseline/",
    filename="model_{}_seed_{}".format(config.model, config.seed))

    trainer = pl.Trainer(
        max_epochs=epoch, logger=wandb_logger
        , strategy="ddp", gpus=1, callbacks=[checkpoint_callback])
    model = FlattenModel(
            token_len=512,
            out_dim=out_size,
            hidden_dim=config.train.hidden_dim,
            dropout_rate=config.train.dropout_rate,
            load_bert=False)
    classification_train = ClassificationTrainer(alpha=config.train.alpha, model=model)
    trainer.fit(classification_train, train_dataloader, validate_dataloader)
    trainer.test(classification_train, test_dataloader)


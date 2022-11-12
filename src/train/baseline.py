from models import FlattenModel, RandomModel, ConvolutionModel, SpecialTokenModel

from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd
import ast

from dataset import SimulateDataset

from trainer import BaselineModelTrainer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader


def baseline_train(data_path, config):
    exp_name = config.name + "_{}_{}".format(config.train.alpha, config.model)
    epoch = config.train.epoch
    debug = config.debug
    gpu_num = torch.cuda.device_count()
    batch_size = config.train.batch_size
    df = pd.read_csv(data_path)
    df["text"] = [ast.literal_eval(d) for d in df["text"]]
    train, validate = train_test_split(df, test_size=0.2)
    validate, test = train_test_split(validate, test_size=0.5)
    print(len(train), len(validate), len(test))
    if debug:
        train = train[: 8 * 2]
        validate = validate[: 8 * 2]
        epoch = 3
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
    save_top_k=1,
    monitor="validation_loss",
    mode="min",
    dirpath="./model/baseline/",
    filename="model_{}_alpha_{}_seed_{}".format(config.model,config.train.alpha, config.seed),
    save_weights_only=True)
    """
    trainer = pl.Trainer(
        max_epochs=epoch, logger=wandb_logger, strategy="ddp", gpus=gpu_num,callbacks=[checkpoint_callback]
    )
    """
    trainer = pl.Trainer(
        max_epochs=epoch, logger=wandb_logger,accelerator = "gpu", callbacks=[checkpoint_callback]
    )

    model = get_model(config)

    modelTrainer = BaselineModelTrainer(alpha=config.train.alpha, model=model)
    trainer.fit(modelTrainer, train_dataloader, validate_dataloader)

    model_path = "model/baseline/model_{}_alpha_{}_seed_{}.ckpt".format(config.model,config.train.alpha, config.seed)
    best_model = modelTrainer.load_from_checkpoint(model_path, alpha=config.train.alpha, model=model)
    predictions = trainer.predict(best_model, test_dataloader)

    eval_with_random(predictions, test, wandb_logger)

def get_model(config):
    if config.model == "flatten":
        model = FlattenModel(
            token_len=512,
            out_dim=config.train.out_dim,
            hidden_dim=config.train.hidden_dim,
            dropout_rate=config.train.dropout_rate,
            load_bert=False,
        )
    if config.model == "special":
        model = SpecialTokenModel(
            token_len=512,
            out_dim=config.train.out_dim,
            hidden_dim=config.train.hidden_dim,
            dropout_rate=config.train.dropout_rate,
            load_bert=False,
        )
    if config.model == "conv":
        model = ConvolutionModel(
            token_len=512,
            out_dim=config.train.out_dim,
            hidden_dim=config.train.hidden_dim,
            dropout_rate=config.train.dropout_rate,
            kernel_size=4,
            stride=2,
            load_bert=False,
        )
    return model

def calc_scores(config, alpha, seed):
    model_path = "model/baseline/model_alpha_{}_seed_{}.ckpt".format(alpha, seed)
    model = get_model(config)
    model_trainer = BaselineModelTrainer.load_from_checkpoint(model_path,alpha=config.train.alpha, model=model)
    df = pd.read_csv("./data/train.csv")
    df["text"] = [ast.literal_eval(d) for d in df["text"]]
    _, validate = train_test_split(df, test_size=0.2)
    _, test = train_test_split(validate, test_size=0.5)
    test_dataset = SimulateDataset(test)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.train.batch_size,
        num_workers=config.dataset.num_workers
    )

    exp_name = "compare_w_random_" + config.name + "_{}_{}".format(config.train.alpha, config.model)
    # loggerの用意
    wandb_logger = WandbLogger(name=exp_name, project="baseline")
    wandb_logger.log_hyperparams(config.train)

    trainer = pl.Trainer(
        logger=wandb_logger,accelerator="gpu"
    )
    predictions = trainer.predict(model_trainer, test_dataloader)
    eval_with_random(predictions, test, wandb_logger)
    wandb_logger.finalize("success")

def eval_with_random(predictions, test, logger):
    size = len(predictions)
    crowd_d = test['cloud_dicision'].to_list()
    system_d = test['system_dicision'].to_list()
    answer = test['correct'].to_list()
    acc, precision, recall, f1, s_count, c_count = 0, 0, 0, 0, 0, 0
    for out in predictions:
        acc += out["test_accuracy"]
        precision += out["test_precision"]
        recall += out["test_recall"]
        f1 += out["test_f1"]
        s_count += out["system_count"]
        c_count += out["crowd_count"]
    acc /= size
    precision /= size
    recall /= size
    f1 /= size
    logger.log_metrics({"test_accuracy": acc})
    logger.log_metrics({"test_precision": precision})
    logger.log_metrics({"test_recall": recall})
    logger.log_metrics({"test_f1": f1})
    logger.log_metrics({"test_system_count": s_count})
    logger.log_metrics({"test_crowd_count": c_count})

    random_pred = RandomModel.predict(system_d, crowd_d, c_count)
    acc = sum([a == r for a, r in zip(answer, random_pred)]) / len(answer)
    precision, recall, f1, _ = precision_recall_fscore_support(random_pred, answer, average="macro")
    logger.log_metrics({"random_accuracy": acc})
    logger.log_metrics({"random_precision": precision})
    logger.log_metrics({"random_recall": recall})
    logger.log_metrics({"random_f1": f1})


from models import FlattenModel, RandomModel, ConvolutionModel, SpecialTokenModel

import pandas as pd
import ast
from tqdm import tqdm

from dataset import SimulateDataset

from trainer import BaselineModelTrainer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, MLFlowLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import precision_recall
from torchmetrics import F1Score


def baseline_train(data_path, config):
    exp_name = config.name + "_{}_{}".format(config.train.alpha, config.model)
    debug = config.debug
    batch_size = config.train.batch_size
    df = pd.read_csv(data_path)
    df["text"] = [ast.literal_eval(d) for d in df["text"]]
    train_df, validate = train_test_split(df, test_size=0.2)
    validate, test = train_test_split(validate, test_size=0.5)
    print(len(train_df), len(validate), len(test))
    if debug:
        train_df = train_df[: 8 * 2]
        validate = validate[: 8 * 2]
        config.train.epoch = 3
    # データセットの用意
    train_df = train_df.reset_index()
    validate = validate.reset_index()
    test = test.reset_index()
    train_dataset = SimulateDataset(train_df)
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
    mlflow_logger = MLFlowLogger(experiment_name = exp_name)
    mlflow_logger.log_hyperparams(config.train)
    # import ipdb;ipdb.set_trace()
    mlflow_logger.log_hyperparams({"mode": config.mode})
    mlflow_logger.log_hyperparams({"seed": config.seed})
    mlflow_logger.log_hyperparams({"model": config.model})
    if config.mode == "train":
        # train(config, wandb_logger, train_dataloader, validate_dataloader)
        train(config, mlflow_logger, train_dataloader, validate_dataloader)
    else:
        # eval(config, test, wandb_logger, test_dataloader)
        eval(config, test, mlflow_logger, test_dataloader)

def train(config, logger, train_dataloader, validate_dataloader):
    gpu_num = torch.cuda.device_count()
    save_path = "./model/baseline/model_{}_alpha_{}_seed_{}.pth".format(config.model,config.train.alpha, config.seed)
    
    trainer = pl.Trainer(
        max_epochs=config.train.epoch, logger=logger, strategy="ddp", gpus=gpu_num,
    )
    model = get_model(config)

    modelTrainer = BaselineModelTrainer(alpha=config.train.alpha, model=model, save_path=save_path, learning_rate = config.train.learning_rate)
    trainer.fit(modelTrainer, train_dataloader, validate_dataloader)
    
def eval(config, test, logger, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(config)
    path = "./model/baseline/model_{}_alpha_{}_seed_{}.pth".format(config.model,config.train.alpha, config.seed)
    model.load_state_dict(torch.load(path))
    model=model.to(device)
    predictions = []
    for batch in tqdm(test_dataloader):
        input_ids = batch["tokens"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        system_dicision = batch["system_dicision"].to(device)
        system_out = batch["system_out"].to(device)
        crowd_dicision = batch["cloud_dicision"].to(device)
        annotator = batch["correct"].to(device)
        start_idx = batch['start_idx'].to(device)
        end_idx = batch['end_idx'].to(device)
        answer = annotator.to("cpu")
        out = model(input_ids, attention_mask,start_idx,end_idx)
        model_ans, s_count, c_count = model.predict(
            out, system_out, system_dicision, crowd_dicision
        )
        acc, precision, recall, f1 = calc_metrics(answer, model_ans)
        predictions += [{
            "test_accuracy": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1.item(),
            "system_count": s_count,
            "crowd_count": c_count,
        }]
    eval_with_random(predictions, test, logger)

def calc_metrics(answer, result):
    f1_score = F1Score()
    acc = sum(answer == result) / len(answer)
    precision, recall = precision_recall(result, answer)
    acc = acc.item()
    precision = precision.item()
    recall = recall.item()
    f1 = f1_score(result, answer)
    return (acc, precision, recall, f1)

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


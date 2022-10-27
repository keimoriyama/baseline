import pandas as pd
import ast

from dataset import SimulateDataset
from model import BaselineModel
from utils import CalculateMetrics

from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from torch.utils.data import DataLoader


seed_everything(1234)
debug = True


def main():
    data_path = "/Users/keimoriyama/Program/lab/data/train.csv"
    df = pd.read_csv(data_path)
    df["text"] = [ast.literal_eval(d) for d in df["text"]]
    df = df.replace(True, 1).replace(False, 0)
    train, validate = train_test_split(df)
    if debug:
        train = train[: 32 * 2]
        validate = validate[: 32 * 2]
    train = train.reset_index()
    validate = validate.reset_index()
    train_dataset = SimulateDataset(train)
    validate_dataset = SimulateDataset(validate)

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    validate_dataloader = DataLoader(validate_dataset, batch_size=32)

    mlflow_logger = pl_loggers.MLFlowLogger()
    trainer = pl.Trainer(max_epochs=1, logger=mlflow_logger)
    Metrics = CalculateMetrics()
    model = BaselineModel(alpha=0.5, metrics=Metrics)
    trainer.fit(
        model,
        train_dataloader,
        validate_dataloader,
    )


if __name__ == "__main__":
    main()

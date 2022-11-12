from pytorch_lightning.utilities.seed import seed_everything

from train import baseline_train, classification_train

from omegaconf import OmegaConf
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", help="hyperparams for exp")
    parser.add_argument("--model", help="choose model to trian and evaluate")
    config = OmegaConf.load("./config/baseline.yml")
    args = parser.parse_args()
    config.train.alpha = float(args.alpha)
    config.model = args.model
    seed_everything(config.seed)
    if config.task == "classification":
        data_path = "./data/classification.csv"
        classification_train(data_path, config)
    elif config.task == "baseline":
        data_path = "./data/train.csv"
        baseline_train(data_path, config)


if __name__ == "__main__":
    main()

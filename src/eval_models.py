from train import calc_scores

from omegaconf import OmegaConf

from pytorch_lightning.utilities.seed import seed_everything

config = OmegaConf.load("./config/baseline.yml")
seed_everything(config.seed)
alphas = [i/10 if i != 0 and i != 10 else i for i in range(0, 11)]
print(alphas)
for a in alphas:
    print(a)
    calc_scores(config, a, config.seed)


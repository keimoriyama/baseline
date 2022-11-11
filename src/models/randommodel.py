import torch
import torch.nn as nn
import random



class RandomModel(nn.Module):
    def __init__(self, out_dim) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, *args):
        return None

    @classmethod
    def predict(cls, system_dicision, crowd_dicision, crowd_count):
        model_ans = []
        crowd_i = random.sample(range(len(crowd_dicision)), crowd_count)
        counts = 0
        for i in range(len(crowd_dicision)):
            if i in crowd_i:
                model_ans.append(crowd_dicision[i])
                counts += 1
            else:
                model_ans.append(system_dicision[i])
        assert counts == crowd_count
        return model_ans

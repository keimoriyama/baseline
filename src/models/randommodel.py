import torch
import torch.nn as nn

class RandomModel(nn.Module):
    def __init__(self,  out_dim) -> None:
        super().__init__()
        self.out_dim = out_dim
    def forward(self, *args):
        return None

    def predict(self, system_dicision, crowd_dicision):
        model_ans = []
        s_count, c_count = 0, 0
        batch_size = len(system_dicision)
        # import ipdb;ipdb.set_trace()
        out = torch.rand(batch_size, self.out_dim)
        for i, o in enumerate(out):
            if o[0] > o[1]:
                model_ans.append(system_dicision[i])
                s_count += 1
            else:
                model_ans.append(crowd_dicision[i])
                c_count += 1
        model_ans = torch.Tensor(model_ans)
        return model_ans, s_count, c_count
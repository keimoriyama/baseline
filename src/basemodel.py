import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, token_len, hidden_dim, out_dim, load_bert,  dropout_rate) -> None:
        super().__init__()
        self.token_len = token_len
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
    def forward(self):
        pass

    def predict(self):
        pass
import torch.nn as nn
import torch

from transformers import RobertaConfig, RobertaModel
import math


class ConvolutionModel(nn.Module):
    def __init__(
        self,
        token_len,
        hidden_dim,
        out_dim,
        dropout_rate,
        kernel_size,
        stride,
        load_bert=False,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.config = RobertaConfig.from_pretrained("./model/config.json")
        if load_bert:
            self.bert = RobertaModel.from_pretrained(
                "./model/pytorch_model.bin", config=self.config
            )
        else:
            self.bert = RobertaModel(config=self.config)
        self.kernel_size = kernel_size
        self.stride = stride
        self.Conv1d = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim // 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.ConvOut = math.ceil(
            (self.config.hidden_size + 2 * 0 - (self.kernel_size - 1) - 1) / self.stride
        )
        # batchsizeが1の時、BatchNormがエラーを吐く
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_dim * 2)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_dim)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.Linear1 = nn.Linear(
            self.ConvOut * self.hidden_dim // 2, self.hidden_dim * 2
        )
        self.Linear2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.Linear3 = nn.Linear(self.hidden_dim, self.out_dim)
        self.params = (
            list(self.Linear1.parameters())
            + list(self.Linear2.parameters())
            + list(self.Linear3.parameters())
            + list(self.Conv1d.parameters())
        )

    def model(self, input):
        batch_size = input.size(0)
        out = self.Conv1d(input)
        out = out.reshape(-1, self.hidden_dim // 2 * self.ConvOut)
        out = self.Linear1(out)
        if batch_size != 1:
            out = self.batch_norm1(out)
        out = self.Linear2(self.dropout(out))
        if batch_size != 1:
            out = self.batch_norm2(out)
        out = self.Linear3(self.dropout(out))
        return out

    def forward(self, input_ids, attention_mask, start_index, end_index):
        out = self.bert(input_ids, attention_mask=attention_mask)
        out = out["last_hidden_state"]
        out = self.model(out)
        return out

    def get_params(self):
        return self.params

import torch
import torch.nn as nn

from transformers import RobertaConfig, RobertaModel

from basemodel import BaseModel


class FlattenModel(nn.Module):
    def __init__(
        self, token_len, hidden_dim, out_dim, dropout_rate, load_bert=False
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

        # batchsizeが1の時、BatchNormがエラーを吐く
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_dim * 2)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.Linear1 = nn.Linear(
            self.token_len * self.config.hidden_size, self.hidden_dim * 2
        )
        self.Linear2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.Linear3 = nn.Linear(self.hidden_dim, self.out_dim)
        self.params = (
            list(self.Linear1.parameters())
            + list(self.Linear2.parameters())
            + list(self.Linear3.parameters())
            + list(self.batch_norm1.parameters())
            + list(self.batch_norm2.parameters())
        )

    def model(self, input):
        # import ipdb;ipdb.set_trace()
        batch_size = input.size(0)
        out = self.Linear1(input)
        if batch_size != 1:
            out = self.batch_norm1(out)
        out = self.Linear2(self.dropout(out))
        if batch_size != 1:
            out = self.batch_norm2(out)
        out = self.Linear3(self.dropout(out))
        return out

    def forward(self, input_ids, attention_mask, start_idx=-1, end_idx=-1):
        out = self.bert(input_ids, attention_mask=attention_mask)
        out = out["last_hidden_state"]
        out = out.reshape(-1, self.token_len * self.config.hidden_size)
        out = self.model(out)
        return out

    def predict(self, out, system_out, system_dicision, crowd_dicision):
        model_ans = []
        s_count, c_count = 0, 0
        for i, (s_out, c_out) in enumerate(zip(system_out, out[:, 1])):
            s_out = s_out.item()
            c_out = c_out.item()
            if s_out > c_out:
                model_ans.append(system_dicision[i])
                s_count += 1
            else:
                model_ans.append(crowd_dicision[i])
                c_count += 1
        model_ans = torch.Tensor(model_ans)
        return model_ans, s_count, c_count

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def get_params(self):
        return self.params

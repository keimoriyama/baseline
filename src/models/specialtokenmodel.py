import torch.nn as nn
import torch
from transformers import RobertaConfig, RobertaModel


class SpecialTokenModel(nn.Module):
    def __init__(
        self, token_len, hidden_dim, out_dim, dropout_rate, load_bert=False
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.config = RobertaConfig.from_pretrained("./model/config.json")
        self.bert = RobertaModel(config=self.config)
        if load_bert:
            self.bert.load_state_dict(torch.load("./model/bert_model.pth"))

        # batchsizeが1の時、BatchNormがエラーを吐く
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_dim * 2)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_dim)
        self.ReLU = nn.ReLU()
        self.Linear1 = nn.Linear(self.config.hidden_size * 2, self.hidden_dim * 2)
        self.Linear2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.Linear3 = nn.Linear(self.hidden_dim, self.out_dim)
        self.params = (
            list(self.Linear1.parameters())
            + list(self.Linear2.parameters())
            + list(self.Linear3.parameters())
            + list(self.batch_norm1.parameters())
            +list(self.batch_norm2.parameters())
        )

    def model(self, input):
        batch_size = input.size(0)
        out = self.ReLU(self.Linear1(input))
        if batch_size != 1:
            out = self.batch_norm1(out)
        out = self.ReLU(self.Linear2(out))
        if batch_size != 1:
            out = self.batch_norm2(out)
        out = self.Linear3(out)
        return out

    def forward(self, input_ids, attention_mask, start_index=-1, end_index=-1):
        out = self.bert(input_ids, attention_mask=attention_mask)
        out = out["last_hidden_state"]
        batch_size = input_ids.size(0)
        # 最初と最後の特殊トークンに当たる特徴量を取得して繋げる
        outputs = [
            torch.cat((out[:, s][i], out[:, e][i]))
            for i, (s, e) in enumerate(zip(start_index, end_index))
        ]
        outputs = torch.stack(outputs)
        # out = out.reshape(-1, self.token_len * self.config.hidden_size)
        out = self.model(outputs)
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
        
    def get_params(self):
        return self.params

import pytorch_lightning as pl
import torch.nn as nn
import torch
from transformers import RobertaConfig, RobertaModel
from torchmetrics.functional import precision_recall
from torchmetrics import F1Score

torch.autograd.set_detect_anomaly(True)

class BaselineModel(pl.LightningModule):
    def __init__(self, alpha,token_len=512, tmp_out_dim = 384, out_dim = 2,load_bert=False, learning_rate= 1e-5, hidden_dim= 512):
        super().__init__()
        self.alpha = alpha
        self.softmax = torch.nn.Softmax(dim=1)
        self.token_len = token_len
        self.config = RobertaConfig.from_pretrained("./model/config.json")
        if load_bert:
            self.bert = RobertaModel.from_pretrained(
                "./model/pytorch_model.bin", config=self.config
            )
        else:
            self.bert = RobertaModel(config=self.config)
        self.tmp_out_dim = tmp_out_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        # self.linear = nn.Linear(self.config.hidden_size, self.tmp_out_dim)
        self.model = nn.Sequential(
            nn.Linear(self.token_len * self.config.hidden_size, self.hidden_dim*2),
            # nn.Linear(self.tmp_out_dim * self.token_len, 1024),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        self.params = list(self.model.parameters()) + list(self.bert.parameters())
        self.f1 = F1Score()
        self.lr = learning_rate
    
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask)
        out = out['last_hidden_state']
        # out = self.linear(out)
        out = out.reshape(-1, self.token_len * self.config.hidden_size)
        # out = out.reshape(-1, self.token_len * self.tmp_out_dim)
        out = self.model(out)
        return out

    def training_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        cloud_out = batch['cloud_dicision']
        system_dicision = batch['system_dicision']
        system_out = batch["system_out"]
        annotator  = batch["correct"]
        out = self.forward(input_ids, attention_mask)
        # モデルの出力に合わせるように学習しているような気がする
        # answer：答えがあっているかどうかの判定（エキスパートの判定）
        # system_out：システムの正誤判定
        # cloud_out：クラウドの正誤判定
        # システムとクラウドのあっている方の正誤判定を採用するようにしたい
        loss = self.loss_function(out, system_out, system_dicision, cloud_out, annotator)
        self.log_dict({"train_loss": loss}, on_epoch=True, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        system_dicision = batch["system_dicision"]
        system_out = batch['system_out']
        cloud_dicision = batch['cloud_dicision']
        annotator  = batch["correct"]
        out = self.forward(input_ids, attention_mask=attention_mask)

        loss = self.loss_function(out, system_out, system_dicision, cloud_dicision, annotator).item()
        model_ans = []
        system_count, cloud_count = 0, 0
        for i, (s_out, c_out) in enumerate(zip(system_out, out[:, 1])):
            s_out = s_out.item()
            c_out = c_out.item()
            if s_out > c_out:
                model_ans.append(system_dicision[i])
                system_count += 1
            else:
                model_ans.append(cloud_dicision[i])
                cloud_count += 1
        model_ans = torch.Tensor(model_ans)
        answer = annotator.to("cpu")
        acc, precision, recall, f1 = self.calc_metrics(answer, model_ans)
        
        log_data = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "validation_loss": loss,
            "model_cloud_count": cloud_count,
            "model_system_count": system_count
        }
        self.log_dict(log_data, on_epoch=True, logger=True)
        return log_data

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.params, lr=self.lr)
        return optimizer

    def loss_function(self, output, system_out, system_dicision, cloud_dicision, annotator):
        # log2(0)が入るのを防ぐために、微小値を足しておく 
        output = torch.stack((system_out, output[:, 1]), -1)
        out = self.softmax(output) + 1e-10
        m1 = (cloud_dicision == annotator).to(int)
        loss =  - (self.alpha * m1 + (1-m1))* torch.log2(out[:, 0]) - m1 * torch.log2(out[:, 1])
        assert not torch.isnan(loss).any()
        loss = torch.mean(loss)
        return loss
    
    def calc_metrics(self,  answer, result):
        acc = sum(answer == result) / len(answer)
        precision, recall = precision_recall(result, answer)
        acc = acc.item()
        precision = precision.item()
        recall = recall.item()
        f1 = self.f1(result, answer)
        return (acc, precision, recall, f1)

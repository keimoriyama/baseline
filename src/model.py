import pytorch_lightning as pl
import torch.nn as nn
import torch
from transformers import RobertaConfig, RobertaModel
from torchmetrics.functional import precision_recall

class BaselineModel(pl.LightningModule):
    def __init__(self, alpha,token_len=512, load_bert=False):
        super().__init__()
        self.alpha = alpha
        self.softmax = torch.nn.Softmax()
        self.token_len = token_len
        self.config = RobertaConfig.from_pretrained("./model/config.json")
        if load_bert:
            self.bert = RobertaModel.from_pretrained(
                "./model/pytorch_model.bin", config=self.config
            )
        else:
            self.bert = RobertaModel(config=self.config)

        self.model = nn.Sequential(
            nn.Linear(self.config.hidden_size * self.token_len, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        system_out = batch["system_dicision"]
        cloud_out = batch["cloud_dicision"]
        answer = batch["correct"]
        out = self.bert(input_ids, attention_mask=attention_mask)
        out = out[0].reshape(-1, self.config.hidden_size * self.token_len)
        out = self.model(out)
        loss = self.loss_function(out, cloud_out)
        self.log_dict({"train_loss": loss}, on_epoch=True, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        system_out = batch["system_dicision"]
        cloud_out = batch["cloud_dicision"]
        answer = batch["correct"]
        out = self.bert(input_ids, attention_mask=attention_mask)
        out = out[0].reshape(-1, self.config.hidden_size * self.token_len)
        out = self.model(out)
        loss = self.loss_function(out, cloud_out).item()
        model_out = out.argmax(1)
        model_ans = []
        for i, out in enumerate(model_out):
            if out == 0:
                model_ans.append(cloud_out[i])
            else:
                model_ans.append(system_out[i])
        model_ans = torch.Tensor(model_ans)
        answer = answer.to("cpu")
        acc, precision, recall, f1 = self.calc_metrics(answer, model_ans)
        log_data = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "validation_loss": loss,
        }
        self.log_dict(log_data, on_epoch=True, on_step=True, logger=True)
        return log_data

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def loss_function(self, output, cloud_out):
        # dimの宣言
        output = self.softmax(output)
        loss = (
            -(self.alpha * cloud_out + (1 - cloud_out)) * output[:, 0]
            - cloud_out * output[:, 1]
        )
        loss = loss.sum() / len(loss)
        return loss
    
    def calc_metrics(self,  answer, result):
        acc = sum(answer == result) / len(answer)
        precision, recall = precision_recall(result, answer)
        acc = acc.item()
        precision = precision.item()
        recall = recall.item()
        f1 = 0
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = precision * recall * 2 / (precision + recall)
        return (acc, precision, recall, f1)

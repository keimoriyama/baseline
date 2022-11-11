import pytorch_lightning as pl
import torch.nn as nn
import torch
from transformers import RobertaConfig, RobertaModel
from torchmetrics.functional import precision_recall
from torchmetrics import F1Score

torch.autograd.set_detect_anomaly(True)


class ClassificationTrainer(pl.LightningModule):
    def __init__(self, alpha, model, learning_rate=1e-5):
        super().__init__()
        self.alpha = alpha
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)
        self.params = self.model.parameters()
        self.f1 = F1Score()
        self.lr = learning_rate
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, start_index, end_index):
        return self.model(input_ids, attention_mask, start_index, end_index)

    def training_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        attribute_id = batch["attribute_id"]
        start_index = batch["start_idx"]
        end_index = batch["end_idx"]
        # import ipdb;ipdb.set_trace()
        out = self.forward(
            input_ids, attention_mask, start_index=start_index, end_index=end_index
        )
        loss = self.loss(out, attribute_id)
        answer = torch.argmax(out, dim=1)
        acc, precision, recall, f1 = self.calc_metrics(answer, attribute_id)
        log_data = {
            "train_loss": loss,
            "train_accuracy": acc,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1,
        }
        self.log_dict(log_data, on_epoch=True, logger=True, batch_size=len(batch))
        return loss

    def calc_metrics(self, answer, result):
        # import ipdb;ipdb.set_trace()
        acc = sum(answer == result) / len(answer)
        precision, recall = precision_recall(result, answer)
        acc = acc.item()
        precision = precision.item()
        recall = recall.item()
        f1 = self.f1(result, answer)
        return (acc, precision, recall, f1)

    def validation_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        attribute_id = batch["attribute_id"]
        start_index = batch["start_idx"]
        end_index = batch["end_idx"]
        out = self.forward(
            input_ids, attention_mask, start_index=start_index, end_index=end_index
        )
        loss = self.loss(out, attribute_id)
        answer = torch.argmax(out, dim=1)
        acc, precision, recall, f1 = self.calc_metrics(answer, attribute_id)
        log_data = {
            "valid_loss": loss,
            "valid_accuracy": acc,
            "valid_precision": precision,
            "valid_recall": recall,
            "valid_f1": f1,
        }
        self.log_dict(log_data, on_epoch=True, logger=True, batch_size=len(batch))
        return

    def test_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        attribute_id = batch["attribute_id"]
        start_index = batch["start_idx"]
        end_index = batch["end_idx"]
        out = self.forward(
            input_ids, attention_mask, start_index=start_index, end_index=end_index
        )
        loss = self.loss(out, attribute_id)
        answer = torch.argmax(out, dim=1)
        acc, precision, recall, f1 = self.calc_metrics(answer, attribute_id)
        log_data = {
            "test_loss": loss,
            "test_accuracy": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        }
        self.log_dict(log_data, on_epoch=True, logger=True, batch_size=len(batch))
        return log_data

    def test_epoch_end(self, output_results):
        size = len(output_results)
        acc, precision, recall, f1 = 0, 0, 0, 0
        for out in output_results:
            acc += out["test_accuracy"]
            precision += out["test_precision"]
            recall += out["test_recall"]
            f1 += out["test_f1"]
        acc /= size
        precision /= size
        recall /= size
        f1 /= size
        self.logger.log_metrics({"test_accuracy": acc})
        self.logger.log_metrics({"test_precision": precision})
        self.logger.log_metrics({"test_recall": recall})
        self.logger.log_metrics({"test_f1": f1})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.get_params(), lr=self.lr)
        return optimizer


class BaselineModel(pl.LightningModule):
    def __init__(self, alpha, model, learning_rate=1e-5):
        super().__init__()
        self.alpha = alpha
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)
        self.params = self.model.parameters()
        self.f1 = F1Score()
        self.lr = learning_rate

    def forward(self, input_ids, attention_mask, start_idx, end_idx):
        return self.model(input_ids, attention_mask,start_idx, end_idx)

    def training_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        crowd_dicision = batch["cloud_dicision"]
        system_dicision = batch["system_dicision"]
        system_out = batch["system_out"]
        annotator = batch["correct"]
        start_idx = batch['start_idx']
        end_idx = batch['end_idx']
        out = self.forward(input_ids, attention_mask, start_idx, end_idx)
        # モデルの出力に合わせるように学習しているような気がする
        # answer：答えがあっているかどうかの判定（エキスパートの判定）
        # system_out：システムの正誤判定
        # cloud_out：クラウドの正誤判定
        # システムとクラウドのあっている方の正誤判定を採用するようにしたい
        loss = self.loss_function(
            out, system_out, system_dicision, crowd_dicision, annotator
        )
        self.log_dict({"train_loss": loss}, on_epoch=True, on_step=True, logger=True)
        model_ans, _, _ = self.model.predict(
            out, system_out, system_dicision, crowd_dicision
        )
        acc, precision, recall, f1 = self.calc_all_metrics(model_ans, annotator)
        log_data = {
            "train_accuracy": acc,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1,
        }
        self.log_dict(log_data, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        system_dicision = batch["system_dicision"]
        system_out = batch["system_out"]
        crowd_dicision = batch["cloud_dicision"]
        annotator = batch["correct"]
        start_idx = batch['start_idx']
        end_idx = batch['end_idx']
        out = self.forward(input_ids, attention_mask,start_idx, end_idx)

        loss = self.loss_function(
            out, system_out, system_dicision, crowd_dicision, annotator
        ).item()

        model_ans, s_count, c_count = self.model.predict(
            out, system_out, system_dicision, crowd_dicision
        )
        acc, precision, recall, f1 = self.calc_all_metrics(model_ans, annotator)
        log_data = {
            "valid_accuracy": acc,
            "valid_precision": precision,
            "valid_recall": recall,
            "valid_f1": f1,
            "validation_loss": loss,
            "system_count": s_count,
            "crowd_count": c_count,
        }
        self.log_dict(log_data, on_epoch=True, logger=True)
        return log_data

    def validation_epoch_end(self, validation_epoch_outputs):
        system_all_count, crowd_all_count = 0, 0
        for out in validation_epoch_outputs:
            system_all_count += out["system_count"]
            crowd_all_count += out["crowd_count"]
        data = {"system_count": system_all_count, "crowd_count": crowd_all_count}
        self.log_dict(data)

    def test_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        system_dicision = batch["system_dicision"]
        system_out = batch["system_out"]
        crowd_dicision = batch["cloud_dicision"]
        annotator = batch["correct"]
        out = self.forward(input_ids, attention_mask=attention_mask)
        if out is not None:
            model_ans, s_count, c_count = self.model.predict(
                out, system_out, system_dicision, crowd_dicision
            )
        else:
            model_ans, s_count, c_count = self.model.predict(
                system_dicision, crowd_dicision
            )
        acc, precision, recall, f1 = self.calc_all_metrics(model_ans, annotator)
        log_data = {
            "test_accuracy": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1.item(),
            "system_count": s_count,
            "crowd_count": c_count,
        }
        return log_data

    def test_epoch_end(self, output_results):
        size = len(output_results)
        acc, precision, recall, f1, s_count, c_count = 0, 0, 0, 0, 0, 0
        for out in output_results:
            acc += out["test_accuracy"]
            precision += out["test_precision"]
            recall += out["test_recall"]
            f1 += out["test_f1"]
            s_count += out["system_count"]
            c_count += out["crowd_count"]
        acc /= size
        precision /= size
        recall /= size
        f1 /= size
        self.logger.log_metrics({"test_accuracy": acc})
        self.logger.log_metrics({"test_precision": precision})
        self.logger.log_metrics({"test_recall": recall})
        self.logger.log_metrics({"test_f1": f1})
        self.logger.log_metrics({"test_system_count": s_count})
        self.logger.log_metrics({"test_crowd_count": c_count})

    def calc_all_metrics(self, model_ans, annotator):
        answer = annotator.to("cpu")
        acc, precision, recall, f1 = self.calc_metrics(answer, model_ans)
        return acc, precision, recall, f1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.get_params(), lr=self.lr)
        return optimizer

    def loss_function(
        self, output, system_out, system_dicision, cloud_dicision, annotator
    ):
        # log2(0)が入るのを防ぐために、微小値を足しておく
        output = torch.stack((system_out, output[:, 1]), -1)
        out = self.softmax(output) + 1e-10
        m1 = (cloud_dicision == annotator).to(int)
        loss = -(self.alpha * m1 + (1 - m1)) * torch.log2(out[:, 0]) - m1 * torch.log2(
            out[:, 1]
        )
        assert not torch.isnan(loss).any()
        loss = torch.mean(loss)
        return loss

    def calc_metrics(self, answer, result):
        acc = sum(answer == result) / len(answer)
        precision, recall = precision_recall(result, answer)
        acc = acc.item()
        precision = precision.item()
        recall = recall.item()
        f1 = self.f1(result, answer)
        return (acc, precision, recall, f1)

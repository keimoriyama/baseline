import torch
from torchmetrics.functional import precision_recall

class CalculateMetrics:
    def __call__(self, answer, result):
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

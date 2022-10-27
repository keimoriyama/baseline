class CalculateMetrics:
    def __call__(self, answer, result):
        acc = sum(answer == result) / len(answer)
        confusion_m = self.calc_confusion_matrix(answer, result)
        precision, recall, f1 = self.calc_precision_recall(confusion_m)
        acc = acc.item()
        print(acc, precision, recall, f1)
        return (acc, precision, recall, f1)

    def calc_confusion_matrix(self, ans, dicision):
        ans = list(ans)
        dicision = list(dicision)
        t = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for answer, system in zip(ans, dicision):
            if answer == 1 and system == 0:
                t["TP"] += 1
            elif answer == 0 and system == 1:
                t["FP"] += 1
            elif answer == 0 and system == 0:
                t["TN"] += 1
            elif answer == 1 and system == 0:
                t["FN"] += 1
        return t

    # デバック用関数
    def calc_precision_recall(self, t):
        precision = t["TP"] / (t["TP"] + t["FP"])
        recall = t["TP"] / (t["TP"] + t["FN"])
        f1 = (2 * precision * recall) / (precision + recall)
        return (precision, recall, f1)

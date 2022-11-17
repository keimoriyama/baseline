import pandas as pd
from tokenizer import JanomeBpeTokenizer

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

import random

from omegaconf import OmegaConf

data_path = "./data/system_df.csv"
tokenizer = JanomeBpeTokenizer("./model/codecs.txt")


def main():
    config = OmegaConf.load("./config/baseline.yml")
    df = pd.read_csv(data_path)
    df = df.reset_index(drop=True)
    df["index_id"] = [i + 1000 for i in range(len(df))]
    df["text_text"] = df["text_text"].apply(remove_return)
    if config.task == "baseline":
        df = df.filter(regex="index_id|system_*|correct|text_text|attribute")
        df = df.fillna(False)
        system = df.filter(regex="index_id|system_*")
        system["system_true_count"] = (system == True).sum(axis=1)
        system["system_false_count"] = (system == False).sum(axis=1)
        system["system_out"] = system["system_true_count"] / (
            system["system_true_count"] + system["system_false_count"]
        )
        df = pd.merge(df, system)
        if config.dataset.name == "worse_system":
            # ワーカーのデータ作成
            threshold = 2
            dicision_df = system["system_true_count"] >= threshold
            # import ipdb;ipdb.set_trace()
            column_name = "crowd_dicision"
            df[column_name] = dicision_df
            # システムのデータ作成
            special_attribute = "所在地"
            threshold = [0, 2]
            data = []
            i = -1
            for _, d in df.iterrows():
                if d["attribute"] == special_attribute:
                    i = threshold[1]
                else:
                    i = threshold[0]
                d["system_dicision"] = d["system_true_count"] > i
                if (
                    (i == threshold[0])
                    and (d["system_dicision"] == d["correct"])
                    and (random.uniform(0, 1) > 0.6)
                ):
                    d["system_dicision"] = not (d["system_dicision"])
                data.append(d)
            worker_df = pd.DataFrame(data)
            worker_df = worker_df[["index_id", "system_dicision"]]
            df = pd.merge(worker_df, df)
        elif config.dataset.name == "only_threshold":
            threthold = [2, 5]
            for t in threthold:
                dicision_df = system["system_true_count"] >= t
                # import ipdb;ipdb.set_trace()
                column_name = ""
                if t == threthold[0]:
                    column_name = "crowd_dicision"
                elif t == threthold[1]:
                    column_name = "system_dicision"
                df[column_name] = dicision_df
            df = pd.merge(df, system)
        df["text"] = df["text_text"].apply(tokenize_text)
        # import ipdb;ipdb.set_trace()
        df = (
            df[
                [
                    "system_dicision",
                    "crowd_dicision",
                    "correct",
                    "text",
                    "attribute",
                    "system_out",
                ]
            ]
            .replace(True, 1)
            .replace(False, 0)
        )
        df.to_csv("./data/train_{}.csv".format(config.dataset.name), index=False)
        calc_metrics(df["correct"], df["system_dicision"])
        calc_metrics(df["correct"], df["crowd_dicision"])

    elif config.task == "classification":
        df = pd.read_csv(data_path, index_col=0)
        attribute_list = df["attribute"].value_counts(normalize=True) * 100 >= 1.0
        attribute_df = pd.DataFrame(attribute_list).reset_index()
        attribute_df = attribute_df.rename(
            columns={"index": "attribute", "attribute": "filter"}
        )
        attribute_df["attribute_id"] = [i for i in range(len(attribute_df))]
        df = pd.merge(df, attribute_df, on="attribute")
        df = df[df["filter"]][["text_text", "attribute", "attribute_id"]]
        df["text"] = df["text_text"].apply(tokenize_text)
        df.to_csv("./data/classification.csv", index=False)


def calc_metrics(ans, out):
    acc = accuracy_score(ans, out)
    pre = precision_score(ans, out)
    recall = recall_score(ans, out)
    f1 = f1_score(ans, out)
    print(
        "accuracy: {:.3}, f1: {:.3}, precision: {:.3}, recall: {:.3}".format(
            acc, f1, pre, recall
        )
    )


def tokenize_text(text):
    return tokenizer.tokenize(text)[0]


def remove_return(s):
    return s.replace("\n", "")


if __name__ == "__main__":
    main()

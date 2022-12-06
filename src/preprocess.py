import pandas as pd
from tokenizer import JanomeBpeTokenizer

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from collections import defaultdict

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
    # import ipdb;ipdb.set_trace()
    if config.task == "baseline":
        df = df.filter(regex="index_id|system_*|correct|text_text|attribute|page_id")
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

        elif config.dataset.name=="artificial_data":
            count = df['attribute'].value_counts()
            c_att, s_att = [], []
            for i, a in enumerate(count.index):
                if i % 2 == 0:
                    c_att.append(a)
                else:
                    s_att.append(a)
            crowd_ans, system_ans = [],[]
            for i in range(len(df)):
                crowd = defaultdict(str)
                d=df.iloc[i]
                attribute = d['attribute']
                crowd['attribute'] = d['attribute']
                crowd['index_id'] = d['index_id']
                #　得意な固有表現についての解答
                if attribute in c_att:
                    if random.uniform(0,1)> 0.8:
                        crowd['crowd_dicision'] = not d['correct']
                    else:
                        crowd['crowd_dicision'] = d['correct']
                # 苦手な固有表現についての解答
                if attribute not in c_att:
                    if random.uniform(0,1)< 0.8:
                        crowd['crowd_dicision'] = not d['correct']
                    else:
                        crowd['crowd_dicision'] = d['correct']
                crowd_ans.append(dict(crowd))
            crowd_df = pd.DataFrame(crowd_ans)

            for i in range(len(df)):
                system = defaultdict(str)
                d=df.iloc[i]
                attribute = d['attribute']
                system['attribute'] = d['attribute']
                system['index_id'] = d['index_id']
                #　得意な固有表現についての解答
                if attribute in s_att:
                    if random.uniform(0,1)> 0.8:
                        system['system_dicision'] = not d['correct']
                    else:
                        system['system_dicision'] = d['correct']
                # 苦手な固有表現についての解答
                if attribute not in s_att:
                    if random.uniform(0,1)< 0.8:
                        system['system_dicision'] = not d['correct']
                    else:
                        system['system_dicision'] = d['correct']
                system_ans.append(dict(system))
            system_df = pd.DataFrame(system_ans)
            # import ipdb;ipdb.set_trace()
            make_df = pd.merge(df, crowd_df, on='index_id')
            df = pd.merge(make_df, system_df, on='index_id')
            import ipdb;ipdb.set_trace()
        df["text"] = df["text_text"].apply(tokenize_text)
        df = (
            df[
                [
                    "page_id",
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

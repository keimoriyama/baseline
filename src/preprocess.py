import pandas as pd
from tokenizer import JanomeBpeTokenizer

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

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
        # import ipdb;ipdb.set_trace()
        system["system_out"] = system["system_true_count"] / (
            system["system_true_count"] + system["system_false_count"]
        )
        threthold = [2,5]
        for t in threthold:
            dicision_df = system["system_true_count"] >= t
            # import ipdb;ipdb.set_trace()
            column_name = ""
            if t == threthold[0]:
                column_name = "crowd_dicision"
            elif t == threthold[1]:
                column_name = "system_dicision"
            df[column_name] = dicision_df
        df["text"] = df["text_text"].apply(tokenize_text)
        df = pd.merge(df, system)
        # import ipdb;ipdb.set_trace()
        df = df[
                [
                    "system_dicision",
                    "crowd_dicision",
                    "correct",
                    "text",
                    "attribute",
                    "system_out",
                ]
            ].replace(True, 1).replace(False, 0)
        df.to_csv("./data/train.csv", index=False)
        print(calc_metrics(df['correct'], df['system_dicision']))
        print(calc_metrics(df['correct'], df['crowd_dicision']))

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
    pre = precision_score(ans,out)
    recall = recall_score(ans, out)
    f1 = f1_score(ans, out)
    return (f1,pre, recall)

def tokenize_text(text):
    return tokenizer.tokenize(text)[0]


def remove_return(s):
    return s.replace("\n", "")


if __name__ == "__main__":
    main()

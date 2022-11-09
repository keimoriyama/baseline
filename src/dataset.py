import torch
from torch.utils.data import Dataset
import json


class ClassificationDataset(Dataset):
    def __init__(self, data, num_tokens=512) -> None:
        super().__init__()
        self.data = self.preprocess(data)
        self.vocab = self.json_load("./model/vocab.json")
        self.num_tokens = num_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        text = d['text']
        attribute = d['attribute']
        attribute_id = d['attribute_id']
        text = ["<s>"] + text + ["</s>"]
        attention_mask = [0] * len(text)
        start_idx = 0
        end_idx = len(text)
        text = self.padding(text, "<pad>", self.num_tokens)
        attention_mask = self.padding(attention_mask, 0, self.num_tokens)

        token_id = [self.vocab.get(token, self.vocab["<unk>"]) for token in text]

        return {
            "tokens": torch.LongTensor(token_id),
            "attention_mask": torch.LongTensor(attention_mask),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "attribute": attribute,
            "attribute_id": attribute_id
        }

    def preprocess(self, data):
        indexes = len(data)
        # import ipdb;ipdb.set_trace()
        data_list = []
        for i in range(indexes):
            d = data.iloc[i].to_dict()
            data_list.append(d)
        return data_list

    def json_load(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    
    def padding(self, array, pad, seq_len):
        if len(array) >= seq_len:
            return array
        return array + [pad] * (seq_len - len(array))


class SimulateDataset(Dataset):
    def __init__(self, data, num_tokens=512):
        self.data = self.preprocess(data)
        self.vocab = self.json_load("./model/vocab.json")
        self.num_tokens = num_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        text = d["text"]
        system_dicision = d["system_dicision"]
        cloud_dicision = d["cloud_dicision"]
        correct = d["correct"]
        system_out = d["system_out"]
        text = ["<s>"] + text + ["</s>"]
        attention_mask = [0] * len(text)

        text = self.padding(text, "<pad>", self.num_tokens)
        attention_mask = self.padding(attention_mask, 0, self.num_tokens)

        token_id = [self.vocab.get(token, self.vocab["<unk>"]) for token in text]
        return {
            "system_dicision": system_dicision,
            "cloud_dicision": cloud_dicision,
            "correct": correct,
            "system_out": system_out,
            "tokens": torch.LongTensor(token_id),
            "attention_mask": torch.LongTensor(attention_mask),
        }

    def preprocess(self, data):
        indexes = len(data)
        data_list = []
        for i in range(indexes):
            d = data.iloc[i].to_dict()
            data_list.append(d)
        return data_list

    def json_load(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def padding(self, array, pad, seq_len):
        if len(array) >= seq_len:
            return array
        return array + [pad] * (seq_len - len(array))

import copy
import pickle
from typing import List, Dict

import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def fasta_to_dict(fasta_path: str) -> Dict[str, str]:
    fasta_dict = {
        record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")
    }
    return fasta_dict


def load_pkl(filename: str) -> Dict[str, List]:
    with open(filename, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


def load_data(feature_name: str, file_path: str) -> Dict[str, List]:
    if feature_name == "bert":
        data_dict = fasta_to_dict(file_path)
        data_dict = prepare_dict_inputs(data_dict)
    else:
        data_dict = load_pkl(file_path)
    return data_dict


class FlexibleDataset(Dataset):
    def __init__(self, data, labels, feature_keys):
        self.data = data

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.feature_keys = feature_keys

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw = self.data[idx]
        i = 0
        out = {}

        if "bert" in self.feature_keys:
            input_ids, mask = raw[i]
            out["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
            out["attention_mask"] = torch.tensor(mask, dtype=torch.long)
            i += 1

        if "unirep" in self.feature_keys:
            out["unirep"] = torch.tensor(raw[i], dtype=torch.float)
            i += 1

        if "esm2" in self.feature_keys:
            out["esm2"] = torch.tensor(raw[i], dtype=torch.float)
            i += 1

        if "prott5" in self.feature_keys:
            out["prott5"] = torch.tensor(raw[i], dtype=torch.float)
            i += 1

        if "esmc" in self.feature_keys:
            out["esmc"] = torch.tensor(raw[i], dtype=torch.float)
            i += 1

        return out, self.labels[idx]


def dataset_file_load(args):
    allowed_features = {"bert", "unirep", "esm2", "prott5", "esmc"}
    for feature in args.feature:
        if feature.lower() not in allowed_features:
            raise ValueError(
                f"Invalid feature '{feature}'. Allowed features are: {', '.join(sorted(allowed_features))}."
            )

    concat_order = ["bert", "unirep", "esm2", "prott5", "esmc"]
    feature_loaders = {
        "bert": args.dataset_path,
        "unirep": args.dataset_path.replace(".fasta", "_unirep.pkl"),
        "esm2": args.dataset_path.replace(".fasta", "_esm2.pkl"),
        "prott5": args.dataset_path.replace(".fasta", "_prott5.pkl"),
        "esmc": args.dataset_path.replace(".fasta", "_esmc.pkl"),
    }
    feature_sizes = {
        "bert": 768,
        "unirep": 1900,
        "esm2": 1280,
        "prott5": 1024,
        "esmc": 1152,
    }
    feature_dicts = []
    feature_size = 0
    for feature_name in concat_order:
        if feature_name in args.feature:
            # 添加每个特征的字典
            feature_dicts.append(load_data(feature_name, feature_loaders[feature_name]))
            feature_size += feature_sizes[feature_name]

    keys = list(feature_dicts[0].keys())

    x, y = [], []

    for key in keys:
        features = []
        for feature_dict in feature_dicts:
            features.append(feature_dict[key])

        x.append(features)
        y.append(1 if key.startswith('amp') else 0)

    return x, y, feature_size


def dataset_load(args):
    x, y, feature_size = dataset_file_load(args)

    if args.mode == "train" or args.mode == "pretrain":
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=args.val_pro, random_state=args.data_random_seed, stratify=y
        )
        train_dataset = FlexibleDataset(x_train, y_train, args.feature)
        val_dataset = FlexibleDataset(x_val, y_val, args.feature)
        train_dl = DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True
        )
        val_dl = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True)
        return train_dl, val_dl, feature_size

    elif args.mode == "test":
        test_dataset = FlexibleDataset(x, y, args.feature)
        test_dl = DataLoader(
            test_dataset, batch_size=args.val_batch_size, shuffle=False
        )
        return test_dl, feature_size

    else:
        raise ValueError("Mode Input Error")


def sequence_to_ids(sequence, unk_token_id=2):
    vocab = {aa: idx + 3 for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    return [vocab.get(aa, unk_token_id) for aa in sequence]


def prepare_dict_inputs(
    seq_dict, cls_token_id=1, sep_token_id=2, pad_token_id=0, max_length=600
):
    result = {}
    for key, seq in seq_dict.items():
        ids = [cls_token_id] + sequence_to_ids(seq) + [sep_token_id]
        if len(ids) > max_length + 2:
            ids = ids[: max_length + 2]
        padding_length = (max_length + 2) - len(ids)
        ids += [pad_token_id] * padding_length
        mask = [1] * (len(ids) - padding_length) + [0] * padding_length
        result[key] = (ids, mask)
    return result

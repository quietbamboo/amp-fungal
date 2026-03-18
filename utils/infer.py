import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils.cal_plm_features import cal_UniRep, cal_ESM2, cal_ProtT5, cal_ESMC
from utils.data import fasta_to_dict, load_data, prepare_dict_inputs
from utils.models import CrossAttentionModel


class InferenceDataset(Dataset):
    def __init__(self, data, feature_keys):
        self.data = data
        self.feature_keys = feature_keys

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key, sequence, raw = self.data[idx]
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

        return key, sequence, out


def fasta_to_features(args):
    fasta_dict = fasta_to_dict(args.dataset_path)

    allowed_features = {"bert", "unirep", "esm2", "prott5", "esmc"}
    for feature in args.feature:
        if feature.lower() not in allowed_features:
            raise ValueError(
                f"Invalid feature '{feature}'. Allowed features are: {', '.join(sorted(allowed_features))}."
            )

    concat_order = ["bert", "unirep", "esm2", "prott5", "esmc"]
    if args.batch_infer:
        feature_loaders = {
            "bert": args.dataset_path,
            "unirep": args.dataset_path.replace(".fasta", "_unirep.pkl"),
            "esm2": args.dataset_path.replace(".fasta", "_esm2.pkl"),
            "prott5": args.dataset_path.replace(".fasta", "_prott5.pkl"),
            "esmc": args.dataset_path.replace(".fasta", "_esmc.pkl"),
        }
    else:
        feature_loaders = {
            "bert": prepare_dict_inputs,
            "unirep": cal_UniRep,
            "esm2": cal_ESM2,
            "prott5": cal_ProtT5,
            "esmc": cal_ESMC,
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
            if args.batch_infer:
                feature_dicts.append(
                    load_data(feature_name, feature_loaders[feature_name])
                )
            else:
                feature_dicts.append(feature_loaders[feature_name](fasta_dict))
            feature_size += feature_sizes[feature_name]

    keys = list(feature_dicts[0].keys())

    x = []

    for key in keys:
        features = []
        for feature_dict in feature_dicts:
            features.append(feature_dict[key])
        x.append((key, fasta_dict[key], features))

    return x, feature_size


def inference(model, data_loader, device):
    model.eval()

    predictions = []
    ids = []
    sequences = []

    with torch.no_grad():
        for keys, seqs, batch_data in tqdm(data_loader, desc="Inference"):
            inputs = {}
            for k, v in batch_data.items():
                inputs[k] = v.to(device)
            outputs = model(**inputs)

            preds = torch.softmax(outputs, dim=-1)
            positive_scores = preds[:, 1]

            ids.extend(keys)
            sequences.extend(seqs)
            predictions.extend(positive_scores.cpu().numpy())

    return ids, sequences, predictions


def infer_uniamp(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Feature with {args.feature}")
    modal_list = args.feature

    data, input_size = fasta_to_features(args)
    dataset = InferenceDataset(data, modal_list)
    data_loader = DataLoader(dataset, batch_size=args.val_batch_size)

    print(f"Build model of input {input_size}")
    if len(modal_list) < 2:
        raise ValueError(
            f"modal_list must contain more than 2 elements, got {len(modal_list)}"
        )

    model = CrossAttentionModel(modal_list, hidden_sizes=[256, 64], output_size=2)

    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path)
    msg = model.load_state_dict(checkpoint)
    print(f"Successfully loaded checkpoint from: {checkpoint_path}")
    print(f"Load status: {msg}")

    model.to(device)
    ids, sequences, predicted_scores = inference(model, data_loader, device)

    df_result = pd.DataFrame(
        {"id": ids, "sequence": sequences, "predicted_score": predicted_scores}
    )

    log_path = args.log
    # 如果是纯文件名（不包含路径分隔符）
    if os.path.basename(log_path) == log_path:
        log_path = os.path.join("./logs", log_path)
    else:
        # 是完整路径，获取目录部分
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    log_path = log_path.replace(".log", ".csv")

    df_result.to_csv(log_path, index=False)
    print("Inference results saved to {}".format(log_path))

import pickle
import re
import os
from typing import Dict, List

import esm2
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
)
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.tokenization import get_esmc_model_tokenizers
from tape import TAPETokenizer
from tape.models.modeling_unirep import UniRepModel
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel


def embed_sequence(client: ESM3ForgeInferenceClient, sequence: str) -> LogitsOutput:
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    if isinstance(protein_tensor, ESMProteinError):
        raise protein_tensor
    output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    return output


def create_temp_save_dir(save_path: str):
    save_dir = os.path.splitext(save_path)[0]
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    return save_dir


# 把临时文件合并
def merge_temp_files(save_path: str, save_dir: str, batch_counter: int) -> Dict[str, List[float]]:
    all_data = {}
    for i in range(batch_counter + 1):
        batch_filename = os.path.join(save_dir, f"batch_{i}.pkl")
        with open(batch_filename, "rb") as f:
            batch_data = pickle.load(f)
            all_data.update(batch_data)

    with open(save_path, "wb") as f:
        pickle.dump(all_data, f)

    for i in range(batch_counter + 1):
        os.remove(os.path.join(save_dir, f"batch_{i}.pkl"))

    # 删除目录
    os.rmdir(save_dir)

    return all_data


def cal_UniRep(
    fasta_dict: Dict[str, str], save_path: str = None, save_size: int = 10000
) -> Dict[str, List[float]]:
    model = UniRepModel.from_pretrained("babbler-1900")
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    tokenizer = TAPETokenizer(vocab="unirep")

    # 创建临时目录
    if save_path:
        save_dir = create_temp_save_dir(save_path)
        batch_counter = 0

    unirep_dict = {}
    for idx, key in enumerate(tqdm(fasta_dict.keys(), desc="Encoding sequences UniRep")):
        # 推理
        token_ids = torch.tensor([tokenizer.encode(fasta_dict[key])]).to(device)
        with torch.no_grad():
            output = model(token_ids)
            sequence_output = output[0]
            avg = sequence_output.mean(dim=1)
        unirep_dict[key] = avg.squeeze().tolist()

        # 保存中间结果
        if save_path and (idx + 1) % save_size == 0:
            batch_filename = os.path.join(save_dir, f"batch_{batch_counter}.pkl")
            with open(batch_filename, "wb") as f:
                pickle.dump(unirep_dict, f)
            unirep_dict.clear()
            batch_counter += 1

    # 保存最后一批结果
    if save_path and unirep_dict:
        final_batch_filename = os.path.join(save_dir, f"batch_{batch_counter}.pkl")
        with open(final_batch_filename, "wb") as f:
            pickle.dump(unirep_dict, f)

    # 合并所有临时文件
    if save_path:
        unirep_dict = merge_temp_files(save_path, save_dir, batch_counter)

    return unirep_dict


def cal_ESM2(
    fasta_dict: Dict[str, str], save_path: str = None, batch_size: int = 2, save_size: int = 10000
) -> Dict[str, List[float]]:
    save_size = save_size // batch_size
    model, alphabet = esm2.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 创建临时目录
    if save_path:
        save_dir = create_temp_save_dir(save_path)
        batch_counter = 0
    
    esm2_dict = {}
    keys = list(fasta_dict.keys())
    for idx, i in enumerate(
        tqdm(range(0, len(keys), batch_size), desc="Encoding sequences ESM2")
    ):
        batch_keys = keys[i : i + batch_size]
        data_batch_ = [(key, fasta_dict[key]) for key in batch_keys]
        batch_labels, batch_strs, batch_tokens = batch_converter(data_batch_)

        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)

        token_representations = results["representations"][33]

        for j, tokens_len in enumerate(batch_lens):
            sequence_representation = token_representations[j, 1 : tokens_len - 1].mean(
                0
            )
            esm2_dict[batch_keys[j]] = sequence_representation.cpu().tolist()

        # 保存中间结果
        if save_path and (idx + 1) % save_size == 0:
            batch_filename = os.path.join(save_dir, f"batch_{batch_counter}.pkl")
            with open(batch_filename, "wb") as f:
                pickle.dump(esm2_dict, f)
            esm2_dict.clear()
            batch_counter += 1

    # 保存最后一批结果
    if save_path and esm2_dict:
        final_batch_filename = os.path.join(save_dir, f"batch_{batch_counter}.pkl")
        with open(final_batch_filename, "wb") as f:
            pickle.dump(esm2_dict, f)

    # 合并所有临时文件
    if save_path:
        esm2_dict = merge_temp_files(save_path, save_dir, batch_counter)

    return esm2_dict


def cal_ProtT5(
    fasta_dict: Dict[str, str], save_path: str = None, batch_size: int = 2, save_size: int = 10000
) -> Dict[str, List[float]]:
    save_size = save_size // batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(
        r"./prott5_pre_models", do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained(r"./prott5_pre_models").to(device)

    # 创建临时目录
    if save_path:
        save_dir = create_temp_save_dir(save_path)
        batch_counter = 0

    prott5_dict = {}
    keys = list(fasta_dict.keys())
    for idx, i in enumerate(
        tqdm(range(0, len(keys), batch_size), desc="Encoding sequences ProtT5")
    ):
        batch_keys = keys[i : i + batch_size]
        data_batch_ = [fasta_dict[key] for key in batch_keys]
        sequence_examples = [
            " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in data_batch_
        ]
        ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        for j in range(len(batch_keys)):
            length_of_sequence = len(fasta_dict[batch_keys[j]])
            embedding = embedding_repr.last_hidden_state[j, :length_of_sequence].mean(
                dim=0
            )
            prott5_dict[batch_keys[j]] = embedding.cpu().tolist()

        # 保存中间结果
        if save_path and (idx + 1) % save_size == 0:
            batch_filename = os.path.join(save_dir, f"batch_{batch_counter}.pkl")
            with open(batch_filename, "wb") as f:
                pickle.dump(prott5_dict, f)
            prott5_dict.clear()
            batch_counter += 1

    # 保存最后一批结果
    if save_path and prott5_dict:
        final_batch_filename = os.path.join(save_dir, f"batch_{batch_counter}.pkl")
        with open(final_batch_filename, "wb") as f:
            pickle.dump(prott5_dict, f)

    # 合并所有临时文件
    if save_path:
        prott5_dict = merge_temp_files(save_path, save_dir, batch_counter)

    return prott5_dict


def cal_ESMC(
    fasta_dict: Dict[str, str], save_path: str = None, save_size: int = 10000
) -> Dict[str, List[float]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # client = ESMC.from_pretrained("esmc_600m").to(device) # or "cpu"
    client = ESMC(
        d_model=1152,
        n_heads=18,
        n_layers=36,
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=True,
    ).eval()
    state_dict = torch.load(
        r"./esmc_600m_2024_12_v0.pth",
        map_location=device,
    )
    client.load_state_dict(state_dict)
    client.to(device)

    # 创建临时目录
    if save_path:
        save_dir = create_temp_save_dir(save_path)
        batch_counter = 0

    esmc_dict = {}

    remaining_fasta_dict = {k: v for k, v in fasta_dict.items() if k not in esmc_dict}

    for idx, (key, seq) in enumerate(
        tqdm(remaining_fasta_dict.items(), desc="Encoding sequences ESMC")
    ):
        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(
            protein_tensor,
            LogitsConfig(
                sequence=True, return_embeddings=True, return_hidden_states=True
            ),
        )
        embeddings = logits_output.embeddings
        embeddings = embeddings.mean(dim=1).squeeze(0).cpu().tolist()
        esmc_dict[key] = embeddings

        # 保存中间结果
        if save_path and (idx + 1) % save_size == 0:
            batch_filename = os.path.join(save_dir, f"batch_{batch_counter}.pkl")
            with open(batch_filename, "wb") as f:
                pickle.dump(esmc_dict, f)
            esmc_dict.clear()
            batch_counter += 1

    # 保存最后一批结果
    if save_path and esmc_dict:
        final_batch_filename = os.path.join(save_dir, f"batch_{batch_counter}.pkl")
        with open(final_batch_filename, "wb") as f:
            pickle.dump(esmc_dict, f)

    # 合并所有临时文件
    if save_path:
        esmc_dict = merge_temp_files(save_path, save_dir, batch_counter)
        
    return esmc_dict

import os
from datetime import datetime

import torch
from tqdm import tqdm

from utils.data import dataset_load
from utils.metrics import compute_metrics
from utils.models import *
from utils.functions import correct_paths

def init_classification_head_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def save_log(log_path, message):
    """
    Args:
        log_path (str): path to save log file
        message (str): log content
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}]\n" + message
    with open(log_path, "a") as f:
        f.write(message)


def train_one_epoch(model, train_dl, optimizer, criterion, device):
    """
    Args:
        model (nn.Module): model to train
        train_dl (DataLoader): training dataloader
        optimizer (torch.optim.Optimizer): optimizer
        criterion (nn.Module): loss function
        args:
    Returns:
        avg_loss (float), metrics tuple
    """
    model.train()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    total_batches = 0

    for batch_data, labels in tqdm(train_dl, desc="Training", leave=True):
        labels = labels.to(device)
        optimizer.zero_grad()

        inputs = {}
        for k, v in batch_data.items():
            inputs[k] = v.to(device)
        outputs = model(**inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds)
        all_labels.append(labels)

        running_loss += loss.item()
        total_batches += 1

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = running_loss / total_batches
    return avg_loss, compute_metrics(all_labels, all_preds)


def validate_one_epoch(model, val_dl, criterion, device):
    """
    Args:
        model (nn.Module): model to validate
        val_dl (DataLoader): validation dataloader
        criterion (nn.Module): loss function
        args:
    Returns:
        avg_loss (float), metrics tuple
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_data, labels in tqdm(val_dl, desc="Validating", leave=False):
            labels = labels.to(device)

            inputs = {}
            for k, v in batch_data.items():
                inputs[k] = v.to(device)
            outputs = model(**inputs)

            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds)
            all_labels.append(labels)

            running_loss += loss.item()
            total_batches += 1

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = running_loss / total_batches
    return avg_loss, compute_metrics(all_labels, all_preds)


def train_process(model, train_dl, val_dl, criterion, optimizer, scheduler, args):
    """
    Args:
        model (nn.Module): model to train
        train_dl (DataLoader): training dataloader
        val_dl (DataLoader): validation dataloader
        criterion (nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        args: arguments containing log_path and patience
        scheduler: Reduce learning rate by a factor when the validation loss does not improve for a certain number of epochs
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    args = correct_paths(args)
    log_path = args.log
    best_model_path = args.save

    args_str = "\n".join(f"{k}: {v}" for k, v in vars(args).items())
    save_log(log_path, args_str + "\n")

    num_epochs = args.epochs
    best_record = None
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_metrics = train_one_epoch(
            model, train_dl, optimizer, criterion, device
        )
        val_loss, val_metrics = validate_one_epoch(
            model, val_dl, criterion, device
        )

        scheduler.step(val_loss)

        log_message = (
            f"Epoch {epoch + 1}/{num_epochs}\n"
            f"Train - Loss: {train_loss:.4f} TP: {train_metrics[0]} FP: {train_metrics[1]} TN: {train_metrics[2]} FN: {train_metrics[3]} Acc: {train_metrics[4]:.4f} Prec: {train_metrics[5]:.4f} Recall: {train_metrics[6]:.4f} F1: {train_metrics[7]:.4f} MCC: {train_metrics[8]:.4f}\n"
            f"Val   - Loss: {val_loss:.4f} TP: {val_metrics[0]} FP: {val_metrics[1]} TN: {val_metrics[2]} FN: {val_metrics[3]} Acc: {val_metrics[4]:.4f} Prec: {val_metrics[5]:.4f} Recall: {val_metrics[6]:.4f} F1: {val_metrics[7]:.4f} MCC: {val_metrics[8]:.4f}\n"
        )

        print(log_message)
        save_log(log_path, log_message)

        current_mcc = val_metrics[8]
        if args.mode == "pretrain":
            torch.save(
                {
                    "feature_extractor": model.feature_extractor.state_dict(),
                    "cnn_lstm_attention": model.cnn_lstm_attention.state_dict(),
                },
                best_model_path.replace(".pth", f"_{epoch + 1}.pth"),
            )

        if args.save_all:
            torch.save(
                model.state_dict(), best_model_path.replace(".pth", f"_{epoch + 1}.pth")
            )

        if args.checkpoint and epoch < 10:
            continue

        if best_record is None or current_mcc > best_record["val_metrics"][8]:
            best_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_metrics": train_metrics,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }
            patience_counter = 0
            if args.mode != "pretrain":
                torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in MCC.")
            break

    summary_message = (
        f"Best Record at Epoch {best_record['epoch']}\n"
        f"Train - Loss: {best_record['train_loss']:.4f} TP: {best_record['train_metrics'][0]} FP: {best_record['train_metrics'][1]} TN: {best_record['train_metrics'][2]} FN: {best_record['train_metrics'][3]} Acc: {best_record['train_metrics'][4]:.4f} Prec: {best_record['train_metrics'][5]:.4f} Recall: {best_record['train_metrics'][6]:.4f} F1: {best_record['train_metrics'][7]:.4f} MCC: {best_record['train_metrics'][8]:.4f}\n"
        f"Val   - Loss: {best_record['val_loss']:.4f} TP: {best_record['val_metrics'][0]} FP: {best_record['val_metrics'][1]} TN: {best_record['val_metrics'][2]} FN: {best_record['val_metrics'][3]} Acc: {best_record['val_metrics'][4]:.4f} Prec: {best_record['val_metrics'][5]:.4f} Recall: {best_record['val_metrics'][6]:.4f} F1: {best_record['val_metrics'][7]:.4f} MCC: {best_record['val_metrics'][8]:.4f}\n"
    )

    print(summary_message)
    save_log(log_path, summary_message)

def train_bert(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.model_random_seed)

    print(f"Feature with {args.feature}")

    train_dl, val_dl, input_size = dataset_load(args)

    print(f"Build model of input {input_size}")
    model = PretrainModel(feature_dim=input_size, hidden_sizes=[256, 64], output_size=2)

    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(f"./checkpoint", args.checkpoint)
        checkpoint = torch.load(checkpoint_path)
        msg = model.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        print(
            f"Successfully loaded feature_extractor checkpoint from: {checkpoint_path}"
        )
        print(f"Load status: {msg}")
        msg = model.cnn_lstm_attention.load_state_dict(checkpoint["cnn_lstm_attention"])
        print(
            f"Successfully loaded cnn_lstm_attention checkpoint from: {checkpoint_path}"
        )
        print(f"Load status: {msg}")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    train_process(model, train_dl, val_dl, criterion, optimizer, scheduler, args)


def train_uniamp(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.model_random_seed)
    print(f"Feature with {args.feature}")

    train_dl, val_dl, input_size = dataset_load(args)

    print(f"Build model of input {input_size}")

    modal_list = args.feature
    if len(modal_list) < 2:
        raise ValueError(
            f"modal_list must contain more than 2 elements, got {len(modal_list)}"
        )

    model = CrossAttentionModel(modal_list, hidden_sizes=[256, 64], output_size=2)

    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join("./checkpoint", args.checkpoint)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: '{args.checkpoint}' or '{checkpoint_path}'")

        checkpoint = torch.load(checkpoint_path)
        msg = model.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        print(
            f"Successfully loaded feature_extractor checkpoint from: {checkpoint_path}"
        )
        print(f"Load status: {msg}")
        msg = model.cnn_lstm_attention.load_state_dict(checkpoint["cnn_lstm_attention"])
        print(
            f"Successfully loaded cnn_lstm_attention checkpoint from: {checkpoint_path}"
        )
        print(f"Load status: {msg}")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    train_process(model, train_dl, val_dl, criterion, optimizer, scheduler, args)

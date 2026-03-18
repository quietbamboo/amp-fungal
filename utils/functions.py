import argparse
from datetime import datetime
import os

def parse_arguments():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='The device to use for computation, e.g., "cuda:0" for the first GPU or "cpu" for CPU only.',
    )

    parser.add_argument(
        "--model",
        type=str,
        default="uniamp",
        choices=["bert", "uniamp"],
        help="The name of the model to be used for training (choices: mlp, bert(for pretraining), uniamp, ml(svm, et)).",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "pretrain", "infer"],
        help='The mode of operation: "train" for training, "test" for evaluation, "pretrain" for pretraining the model, "infer" for inference".',
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=r"./data/amp/training_dataset.fasta",
        help="Path to the dataset file (*.fasta). This dataset will be used for training or test.",
    )

    parser.add_argument(
        "--feature",
        type=str,
        nargs="+",
        default=["bert", "unirep", "esm2", "prott5"],
        choices=["bert", "unirep", "esm2", "prott5", "esmc"],
        help="Feature(s) used for the model. Choices: bert, unirep, esm2, prott5, esmc. "
        "Multiple features can be combined by specifying them separated by spaces, "
        'e.g., "--feature unirep esmc".',
    )

    parser.add_argument(
        "--save",
        type=str,
        default=f"{timestamp}_model.pth",
        help="Filename to save the trained model (default: uses timestamp).",
    )

    parser.add_argument(
        "--log",
        type=str,
        default=f"{timestamp}_train.log",
        help="Filename to save the training logs (default: uses timestamp).",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename (will be loaded from ./checkpoint/xxx). Default is None.",
    )

    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Save the model at every epoch (default: only save the best model/False)",
    )

    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for the optimizer."
    )

    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Number of epochs with no improvement after which training will stop (early stopping).",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=128, help="Batch size for training."
    )

    parser.add_argument(
        "--val_batch_size", type=int, default=128, help="Batch size for validation."
    )

    parser.add_argument(
        "--data_random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility, used for splitting the dataset into training and validation sets.",
    )

    parser.add_argument(
        "--model_random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility, used for model parameter initialization.",
    )

    parser.add_argument(
        "--val_pro",
        type=float,
        default=0.2,
        help="Proportion of the dataset to be used for validation (default: 20%).",
    )

    parser.add_argument(
        "--batch_infer",
        action="store_true",
        help="If True, Inferred .pkl files will be used for batch inference",
    )

    return parser.parse_args()


def ensure_dir_exists(dir_path):
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def correct_paths(args):
    if args.log:
        log_path = args.log
        if os.path.basename(log_path) == log_path:
            log_path = os.path.join("./logs", log_path)
        else:
            log_dir = os.path.dirname(log_path)
            ensure_dir_exists(log_dir)
        args.log = log_path

    if args.save:
        save_path = args.save
        filename = os.path.basename(save_path)
        name_without_ext = os.path.splitext(filename)[0]

        if args.mode == "pretrain":
            base_dir = "./checkpoint"
            target_dir = os.path.join(base_dir, name_without_ext)
            ensure_dir_exists(target_dir)
            args.save = os.path.join(target_dir, filename)
        elif args.save_all:
            if os.path.dirname(save_path) == "" or os.path.dirname(save_path) == ".":
                base_dir = "./models"
                target_dir = os.path.join(base_dir, name_without_ext)
            else:
                base_dir = os.path.dirname(save_path)
                target_dir = os.path.join(base_dir, name_without_ext)
            ensure_dir_exists(target_dir)
            args.save = os.path.join(target_dir, filename)
        else:
            if os.path.basename(save_path) == save_path:
                save_path = os.path.join("./models", save_path)
                ensure_dir_exists("./models")
                args.save = save_path
            else:
                save_dir = os.path.dirname(save_path)
                ensure_dir_exists(save_dir)
                args.save = save_path
    return args
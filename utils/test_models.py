import os

from utils.data import dataset_load
from utils.models import *
from utils.train_models import validate_one_epoch


def test_process(model, test_dl, criterion, device):
    """
    Args:
        model (nn.Module): model to train
        test_dl (DataLoader): test dataloader
        criterion (nn.Module): loss function
        args: arguments containing log_path and patience
    """
    test_loss, test_metrics = validate_one_epoch(
        model, test_dl, criterion, device
    )

    log_message = f"Test   - Loss: {test_loss:.4f} TP: {test_metrics[0]} FP: {test_metrics[1]} TN: {test_metrics[2]} FN: {test_metrics[3]} Acc: {test_metrics[4]:.4f} Prec: {test_metrics[5]:.4f} Recall: {test_metrics[6]:.4f} F1: {test_metrics[7]:.4f} MCC: {test_metrics[8]:.4f}\n"
    print(log_message)


def test_uniamp(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Feature with {args.feature}")
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join("./models", args.checkpoint)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: '{args.checkpoint}' or '{checkpoint_path}'")
    else:
        raise ValueError("--checkpoint is required. Please provide a checkpoint path.")

    test_dl, input_size = dataset_load(args)

    print(f"Build model of input {input_size}")
    modal_list = args.feature
    if len(modal_list) < 2:
        raise ValueError(
            f"modal_list must contain more than 2 elements, got {len(modal_list)}"
        )
    model = CrossAttentionModel(modal_list, hidden_sizes=[256, 64], output_size=2)

    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    test_process(model, test_dl, criterion, device)

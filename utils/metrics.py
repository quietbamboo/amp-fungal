import torch
import torch.nn as nn


def compute_metrics(y_true, y_pred):
    """
    Args:
        y_true (Tensor): shape (num_samples,), ground truth labels (0 or 1)
        y_pred (Tensor): shape (num_samples,), predicted labels (0 or 1)
    Returns:
        Tuple of tp, fp, tn, fn, accuracy, precision, recall, f1, mcc
    """
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    total = tp + fp + tn + fn + 1e-8
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + 1e-8
    mcc = mcc_numerator / mcc_denominator

    return tp, fp, tn, fn, accuracy, precision, recall, f1, mcc

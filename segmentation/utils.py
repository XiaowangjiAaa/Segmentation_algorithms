from typing import Dict

import torch
import torch.nn.functional as F


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Compute mIoU, accuracy and F1-score."""
    pred_labels = pred.argmax(dim=1)
    target = target.squeeze(1)
    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)
    tp = torch.zeros(num_classes, dtype=torch.float64)
    fp = torch.zeros(num_classes, dtype=torch.float64)
    fn = torch.zeros(num_classes, dtype=torch.float64)
    for cls in range(num_classes):
        pred_mask = pred_labels == cls
        target_mask = target == cls
        intersection[cls] = (pred_mask & target_mask).sum()
        union[cls] = (pred_mask | target_mask).sum()
        tp[cls] = intersection[cls]
        fp[cls] = pred_mask.sum() - tp[cls]
        fn[cls] = target_mask.sum() - tp[cls]
    iou = (intersection / union.clamp(min=1)).mean().item()
    accuracy = (pred_labels == target).float().mean().item()
    precision = (tp / (tp + fp).clamp(min=1)).mean().item()
    recall = (tp / (tp + fn).clamp(min=1)).mean().item()
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"miou": iou, "accuracy": accuracy, "f1": f1}

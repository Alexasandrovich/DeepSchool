from typing import Dict

import torch
from evaluate import load as load_metric
from torch import nn
from transformers import SegformerForSemanticSegmentation


def init_model_with_pretrain(id2label: Dict, label2id: Dict, pretrain_path: str = None):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    if pretrain_path:
        model = torch.load(pretrain_path, map_location="cpu")["model"]

    return model


def evaluate_model(model, valid_dataloader, id2label):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = load_metric("mean_iou")
    predictions = []
    references = []

    with torch.no_grad():
        model.eval()
        for batch in valid_dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(pixel_values=pixel_values, labels=labels)
            upsampled_logits = nn.functional.interpolate(
                logits.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1)
            predictions.append(predicted.detach().cpu().numpy())
            references.append(labels.detach().cpu().numpy())

        # note that the metric expects predictions + labels as numpy arrays

    metrics = metric._compute(
        predictions=predictions,
        references=references,
        num_labels=len(id2label),
        ignore_index=255,
        reduce_labels=False,  # we've already reduced the labels before)
    )

    print("Mean_iou:", metrics["mean_iou"])
    print("Mean accuracy:", metrics["mean_accuracy"])

    return metrics

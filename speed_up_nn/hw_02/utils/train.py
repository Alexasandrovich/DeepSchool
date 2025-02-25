from dataclasses import dataclass
from .data import init_dataloaders
import torch
import typing as tp
from torch import nn
from tqdm.auto import tqdm
from torch.nn import functional as F
from .model import evaluate_model
from transformers import get_linear_schedule_with_warmup

@dataclass
class TrainParams:
    n_epochs: int
    lr: float
    batch_size: int
    n_workers: int
    device: torch.device
    temperature: float
    intermediate_layers_weights: tp.Tuple[float, float, float, float]
    loss_weight: float
    last_layer_loss_weight: float
    intermediate_attn_layers_weights: tp.Tuple[float, float, float, float]
    intermediate_feat_layers_weights: tp.Tuple[float, float, float, float]
    warmup_steps: int


mse_loss = nn.MSELoss()
kl_loss = nn.KLDivLoss()
id2label = {
    0: "background",
    1: "human",
}


class TeacherToStudentAdapter(nn.Module):
    # we make an adapter from teacher to student, as it is easier to throw away the excess than to invent something else
    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(teacher_dim, student_dim),
            nn.ReLU()
        )

    def forward(self, teacher_features: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = teacher_features.shape
        teacher_features = teacher_features.permute(0,2,3,1).contiguous()  # [B, H, W, C]
        teacher_features = teacher_features.view(-1, channels)  # [B*H*W, C]
        adapted = self.adapter(teacher_features)  # [B*H*W, student_dim]
        adapted = adapted.view(batch, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return adapted

adapters = nn.ModuleList([
    TeacherToStudentAdapter(teacher_dim=32, student_dim=16),
    TeacherToStudentAdapter(teacher_dim=64, student_dim=32),
    TeacherToStudentAdapter(teacher_dim=160, student_dim=80),
    TeacherToStudentAdapter(teacher_dim=256, student_dim=128)
]).to('cuda')

def calc_last_layer_loss(student_logits, teacher_logits, weight, temperature=1.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    return kl_loss(student_log_probs, teacher_probs) * weight


def calc_intermediate_layers_feat_loss(student_feats, teacher_feats, weights):
    total_loss = 0.0
    for i, student_feat in enumerate(student_feats):
        teacher_feat = teacher_feats[i]

        # need adapter after pruning
        adapted_teacher_feat = adapters[i](teacher_feat)
        loss = mse_loss(student_feat, adapted_teacher_feat)
        weight_value = weights[i] if i < len(weights) else 1.0

        total_loss += weight_value * loss
    return total_loss


def calc_intermediate_layers_attn_loss(student_attentions, teacher_attentions, weights,
                                       student_teacher_attention_mapping):
    intermediate_kl_loss = 0.0
    for student_idx, teacher_idx in student_teacher_attention_mapping.items():
        student_attention_block = student_attentions[student_idx]
        teacher_attention_block = teacher_attentions[teacher_idx]

        # attention blocks already transfers due to softmax
        student_probs = torch.log(student_attention_block + 1e-9)
        teacher_probs = teacher_attention_block

        intermediate_kl_loss += kl_loss(student_probs, teacher_probs) * weights[student_idx]

    return intermediate_kl_loss

def train(
        teacher_model,
        student_model,
        train_params: TrainParams,
        student_teacher_attention_mapping,
        tb_writer,
        save_dir,
):
    teacher_model.to(train_params.device)
    student_model.to(train_params.device)

    teacher_model.eval()

    train_dataloader, valid_dataloader = init_dataloaders(
        root_dir=".",
        batch_size=train_params.batch_size,
        num_workers=train_params.n_workers,
    )

    optimizer = torch.optim.AdamW(
        list(student_model.parameters()) + list(adapters.parameters()),
        lr=train_params.lr
    )
    # warmup need after pruning
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_params.warmup_steps,
        num_training_steps=len(train_dataloader) * train_params.n_epochs
    )

    step = 0
    for epoch in range(train_params.n_epochs):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for idx, batch in pbar:
            student_model.train()
            # get the inputs;
            pixel_values = batch['pixel_values'].to(train_params.device)
            labels = batch['labels'].to(train_params.device)

            optimizer.zero_grad()

            # forward + backward + optimize
            student_outputs = student_model(
                pixel_values=pixel_values,
                labels=labels,
                output_attentions=True,
                output_hidden_states=True,
            )
            loss, student_logits = student_outputs.loss, student_outputs.logits

            # Чего это мы no_grad() при тренировке поставили?!
            # Freeze?
            with torch.no_grad():
                teacher_output = teacher_model(
                    pixel_values=pixel_values,
                    labels=labels,
                    output_attentions=True,
                    output_hidden_states=True,
                )

            last_layer_loss = calc_last_layer_loss(
                student_logits,
                teacher_output.logits,
                train_params.last_layer_loss_weight,
                train_params.temperature
            )

            student_attentions, teacher_attentions = student_outputs.attentions, teacher_output.attentions
            student_hidden_states, teacher_hidden_states = student_outputs.hidden_states, teacher_output.hidden_states

            intermediate_layer_att_loss = calc_intermediate_layers_attn_loss(
                student_attentions,
                teacher_attentions,
                train_params.intermediate_attn_layers_weights,
                student_teacher_attention_mapping,
            )

            intermediate_layer_feat_loss = calc_intermediate_layers_feat_loss(
                student_hidden_states,
                teacher_hidden_states,
                train_params.intermediate_feat_layers_weights,
            )

            total_loss = loss * train_params.loss_weight + last_layer_loss
            if intermediate_layer_att_loss is not None:
                total_loss += intermediate_layer_att_loss

            if intermediate_layer_feat_loss is not None:
                total_loss += intermediate_layer_feat_loss

            step += 1

            total_loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(f'total loss: {total_loss.item():.3f}')

            for loss_value, loss_name in (
                    (loss, 'loss'),
                    (total_loss, 'total_loss'),
                    (last_layer_loss, 'last_layer_loss'),
                    (intermediate_layer_att_loss, 'intermediate_layer_att_loss'),
                    (intermediate_layer_feat_loss, 'intermediate_layer_feat_loss'),
            ):
                if loss_value is None:  # для выключенной дистилляции атеншенов
                    continue
                tb_writer.add_scalar(
                    tag=loss_name,
                    scalar_value=loss_value.item(),
                    global_step=step,
                )

        # после модификаций модели обязательно сохраняйте ее целиком, чтобы подгрузить ее в случае чего
        torch.save(
            {
                'model': student_model,
                'state_dict': student_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            },
            f'{save_dir}/ckpt_{epoch}.pth',
        )

        eval_metrics = evaluate_model(student_model, valid_dataloader, id2label)

        for metric_key, metric_value in eval_metrics.items():
            if not isinstance(metric_value, float):
                continue
            tb_writer.add_scalar(
                tag=f'eval_{metric_key}',
                scalar_value=metric_value,
                global_step=epoch,
            )

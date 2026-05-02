"""Training and evaluation entry points."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from .dataset import filter_for_task, load_corpus
from .metrics import (
    compute_task_metrics,
    mechanism_metrics_for_batch,
    soft_assignment_to_labels,
    stability_from_partitions,
)
from .models import create_model, model_family
from .models.common import task_loss, task_target
from .models.factory import calibrate_hidden_dim
from .utils import save_json, set_global_seed


AUX_LOSS_WEIGHTS = {
    "gin": {},
    "gatv2": {},
    "graphgps": {},
    "diffpool": {"link_loss": 1.0, "entropy_loss": 0.1},
    "mincut": {"mincut_loss": 1.0, "orthogonality_loss": 1.0},
    "ptr": {
        "cut_loss": 1.0,
        "orthogonality_loss": 1.0,
        "entropy_loss": 0.1,
        "coarse_edge_consistency_loss": 0.5,
    },
    "ptr_sup": {
        "cut_loss": 1.0,
        "orthogonality_loss": 1.0,
        "entropy_loss": 0.1,
        "coarse_edge_consistency_loss": 0.5,
        "assignment_supervision_loss": 1.0,
    },
}


def primary_metric_name(task: str) -> str:
    return {
        "topology": "macro_f1",
        "bridge_count": "mae",
        "attack_disconnect": "auroc",
        "redundant": "auroc",
    }[task]


def is_metric_better(task: str, candidate: float, incumbent: float | None) -> bool:
    if incumbent is None:
        return True
    if task == "bridge_count":
        return candidate < incumbent
    return candidate > incumbent


def subset_data(data_list, limit: int | None, seed: int) -> list:
    if limit is None or limit >= len(data_list):
        return list(data_list)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data_list))[:limit]
    return [data_list[int(index)] for index in indices]


def build_loader(data_list, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def add_auxiliary_losses(model_name: str, main_loss: torch.Tensor, loss_terms: dict[str, torch.Tensor]) -> torch.Tensor:
    total = main_loss
    for key, weight in AUX_LOSS_WEIGHTS[model_name].items():
        if key in loss_terms:
            total = total + weight * loss_terms[key]
    return total


def evaluate(
    model,
    loader: DataLoader,
    task: str,
    device: torch.device,
    collect_mechanisms: bool = False,
) -> dict:
    model.eval()
    logits_all = []
    target_all = []
    mechanism_values = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch, return_aux=collect_mechanisms)
            target = task_target(batch, task)
            logits_all.append(outputs.logits.detach().cpu())
            target_all.append(target.detach().cpu())
            if collect_mechanisms and task == "topology":
                batch_scores = mechanism_metrics_for_batch(outputs, batch)
                for key, value in batch_scores.items():
                    mechanism_values[key].append(value)
    metrics = compute_task_metrics(task, torch.cat(logits_all), torch.cat(target_all))
    if mechanism_values:
        metrics.update({key: float(np.mean(values)) for key, values in mechanism_values.items()})
    return metrics


def evaluate_stability(model, data_list, device: torch.device) -> dict[str, float]:
    groups = defaultdict(list)
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch, return_aux=True)
            graph = batch.to_data_list()[0]
            if outputs.assignments is None:
                continue
            labels = soft_assignment_to_labels(outputs.assignments[0])
            groups[int(graph.meta["stability_group"])].append(labels)
    scores = [stability_from_partitions(partitions) for partitions in groups.values() if len(partitions) >= 2]
    return {"stability_nmi": float(np.mean(scores)) if scores else 0.0}


def run_experiment(
    *,
    data_root: str,
    output_root: str,
    model_name: str,
    task: str,
    seed: int,
    train_limit: int | None = None,
    val_limit: int | None = None,
    eval_limits: dict[str, int | None] | None = None,
    min_params: int = 405_000,
    max_params: int = 495_000,
    lr: float = 3.0e-4,
    weight_decay: float = 1.0e-5,
    max_epochs: int = 200,
    patience: int = 30,
    batch_size_flat: int = 32,
    batch_size_hier: int = 16,
) -> dict:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = load_corpus(data_root)
    train_data = filter_for_task(subset_data(corpus["train"], train_limit, seed), task)
    val_data = filter_for_task(subset_data(corpus["val"], val_limit, seed + 1), task)
    input_dim = int(train_data[0].x.size(-1))
    hidden_dim, num_params, num_layers = calibrate_hidden_dim(
        model_name,
        input_dim,
        task,
        min_params,
        max_params,
    )
    model = create_model(
        model_name,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        task=task,
        num_layers=num_layers,
    )
    model.to(device)

    batch_size = batch_size_flat if model_family(model_name) == "direct" else batch_size_hier
    train_loader = build_loader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = build_loader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_metric = None
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch, return_aux=False)
            target = task_target(batch, task)
            main_loss = task_loss(outputs.logits, target, task)
            loss = add_auxiliary_losses(model_name, main_loss, outputs.loss_terms)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate(
            model,
            val_loader,
            task=task,
            device=device,
            collect_mechanisms=(task == "topology"),
        )
        current_primary = val_metrics[primary_metric_name(task)]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                **val_metrics,
            }
        )
        if is_metric_better(task, current_primary, best_metric):
            best_metric = current_primary
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("training failed to produce a checkpoint")
    model.load_state_dict(best_state)

    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / f"{model_name}_{task}_seed{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "hidden_dim": hidden_dim}, checkpoint_path)

    eval_limits = eval_limits or {}
    eval_summaries = {}
    for split_name in ["id_test", "near_ood", "far_ood", "cell_size_ood"]:
        if split_name not in corpus:
            continue
        split_data = filter_for_task(
            subset_data(corpus[split_name], eval_limits.get(split_name), seed + 7),
            task,
        )
        split_loader = build_loader(split_data, batch_size=batch_size, shuffle=False)
        eval_summaries[split_name] = evaluate(
            model,
            split_loader,
            task=task,
            device=device,
            collect_mechanisms=(task == "topology"),
        )

    if task == "topology" and "stability" in corpus:
        eval_summaries["stability"] = evaluate_stability(model, corpus["stability"], device)

    summary = {
        "model": model_name,
        "task": task,
        "seed": seed,
        "device": str(device),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_params": num_params,
        "best_epoch": best_epoch,
        "history": history,
        "eval": eval_summaries,
    }
    save_json(summary, output_path / f"{model_name}_{task}_seed{seed}.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bridge_v1 model.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--id-test-limit", type=int, default=None)
    parser.add_argument("--near-ood-limit", type=int, default=None)
    parser.add_argument("--far-ood-limit", type=int, default=None)
    parser.add_argument("--cell-size-ood-limit", type=int, default=None)
    parser.add_argument("--min-params", type=int, default=405000)
    parser.add_argument("--max-params", type=int, default=495000)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size-flat", type=int, default=32)
    parser.add_argument("--batch-size-hier", type=int, default=16)
    args = parser.parse_args()

    summary = run_experiment(
        data_root=args.data_root,
        output_root=args.output_root,
        model_name=args.model,
        task=args.task,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        eval_limits={
            "id_test": args.id_test_limit,
            "near_ood": args.near_ood_limit,
            "far_ood": args.far_ood_limit,
            "cell_size_ood": args.cell_size_ood_limit,
        },
        min_params=args.min_params,
        max_params=args.max_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size_flat=args.batch_size_flat,
        batch_size_hier=args.batch_size_hier,
    )
    print(summary)


if __name__ == "__main__":
    main()

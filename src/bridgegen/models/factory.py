"""Model construction and hidden-dimension calibration."""

from __future__ import annotations

from .common import count_parameters
from .flat import GATv2Model, GINModel, GraphGPSModel
from .hierarchical import DiffPoolModel, MinCutPoolModel, PerceptionThenReasoningModel


MODEL_FAMILIES = {
    "gin": "direct",
    "gatv2": "direct",
    "graphgps": "direct",
    "diffpool": "pooling",
    "mincut": "pooling",
    "ptr": "perception",
    "ptr_sup": "perception",
}


def model_family(model_name: str) -> str:
    return MODEL_FAMILIES[model_name]


def create_model(
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    task: str,
    num_layers: int = 4,
):
    if model_name == "gin":
        return GINModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, task=task)
    if model_name == "gatv2":
        return GATv2Model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            task=task,
        )
    if model_name == "graphgps":
        return GraphGPSModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            task=task,
        )
    if model_name == "diffpool":
        return DiffPoolModel(input_dim=input_dim, hidden_dim=hidden_dim, task=task)
    if model_name == "mincut":
        return MinCutPoolModel(input_dim=input_dim, hidden_dim=hidden_dim, task=task)
    if model_name == "ptr":
        return PerceptionThenReasoningModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            task=task,
            supervised_cells=False,
        )
    if model_name == "ptr_sup":
        return PerceptionThenReasoningModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            task=task,
            supervised_cells=True,
        )
    raise ValueError(f"unsupported model: {model_name}")


def calibrate_hidden_dim(
    model_name: str,
    input_dim: int,
    task: str,
    min_params: int,
    max_params: int,
) -> tuple[int, int, int]:
    candidates = [64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 288, 320, 352, 384]
    layer_candidates = [4, 5, 6] if model_name in {"gin", "gatv2", "graphgps"} else [4]
    nearest = None
    nearest_distance = 10**18
    for num_layers in layer_candidates:
        for hidden_dim in candidates:
            model = create_model(
                model_name,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                task=task,
                num_layers=num_layers,
            )
            num_params = count_parameters(model)
            if min_params <= num_params <= max_params:
                return hidden_dim, num_params, num_layers
            center = (min_params + max_params) // 2
            distance = abs(num_params - center)
            if distance < nearest_distance:
                nearest = (hidden_dim, num_params, num_layers)
                nearest_distance = distance
    if nearest is None:
        raise RuntimeError("failed to calibrate hidden dimension")
    return nearest

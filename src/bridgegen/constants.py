"""Project constants."""

TASK_NAMES = (
    "topology",
    "bridge_count",
    "attack_disconnect",
    "redundant",
)

TEMPLATE_NAMES = (
    "barbell",
    "bow_tie",
    "multi_neck",
    "redundant_bow_tie",
)

TOPOLOGY_TO_INDEX = {name: idx for idx, name in enumerate(TEMPLATE_NAMES)}
INDEX_TO_TOPOLOGY = {idx: name for name, idx in TOPOLOGY_TO_INDEX.items()}

ROLE_NAMES = (
    "wing",
    "bridge",
    "transit",
    "distractor",
)

ROLE_TO_INDEX = {name: idx for idx, name in enumerate(ROLE_NAMES)}
INDEX_TO_ROLE = {idx: name for name, idx in ROLE_TO_INDEX.items()}

PRIMITIVE_NAMES = (
    "er",
    "watts_strogatz",
    "random_regular",
    "motif_bundle",
)

TASK_TO_DEFAULT_METRIC = {
    "topology": "macro_f1",
    "bridge_count": "mae",
    "attack_disconnect": "auroc",
    "redundant": "auroc",
}


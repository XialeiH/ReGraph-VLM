#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


METRICS = ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize lightweight publication-readiness result artifacts cited by may30.tex."
    )
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument(
        "--external-summary-dir",
        type=Path,
        default=Path("external_validation/summary"),
    )
    parser.add_argument("--source-tex", default="reports/neurips_report/may30.tex")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    names = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row(source: str, model: str, n: int, values: dict[str, tuple[float, float | None]]) -> dict[str, object]:
    out: dict[str, object] = {"model": model, "n": n, "source": source}
    for metric in METRICS:
        mean, std = values.get(metric, ("", ""))  # type: ignore[arg-type]
        out[f"{metric}_mean"] = mean
        out[f"{metric}_std"] = "" if std is None else std
    return out


def write_main_tables(final: Path, tex: str) -> None:
    allfold_source = f"{tex}: Table tab:cross_subject_main; canonical summary table_allfold_final_summary.csv"
    write_csv(
        final / "table_allfold_final.csv",
        [
            row(
                allfold_source,
                "ROI-MLP+CLIP",
                24,
                {
                    "AUROC": (0.8164, 0.0518),
                    "AUPRC": (0.7896, 0.0532),
                    "R@5": (0.0782, 0.0273),
                    "MRR": (0.0636, 0.0190),
                    "image_R@5": (0.0729, 0.0280),
                    "brain_R@5": (0.0837, 0.0291),
                    "brain_MRR": (0.0668, 0.0193),
                },
            ),
            row(
                allfold_source,
                "Flat ReGraph+CLIP",
                24,
                {
                    "AUROC": (0.8210, 0.0481),
                    "AUPRC": (0.8022, 0.0460),
                    "R@5": (0.0865, 0.0318),
                    "MRR": (0.0677, 0.0213),
                    "image_R@5": (0.0812, 0.0305),
                    "brain_R@5": (0.0966, 0.0299),
                    "brain_MRR": (0.0749, 0.0189),
                },
            ),
            row(
                allfold_source,
                "Gated ReGraph/BNT+CLIP",
                24,
                {
                    "AUROC": (0.8259, 0.0523),
                    "AUPRC": (0.8065, 0.0528),
                    "R@5": (0.0899, 0.0357),
                    "MRR": (0.0695, 0.0240),
                    "image_R@5": (0.0847, 0.0318),
                    "brain_R@5": (0.0996, 0.0310),
                    "brain_MRR": (0.0773, 0.0207),
                },
            ),
        ],
    )

    heldout_source = f"{tex}: Table tab:heldout"
    write_csv(
        final / "table_heldout_image.csv",
        [
            row(heldout_source, "ROI-MLP+CLIP", 24, {"AUROC": (0.8097, 0.0547), "AUPRC": (0.7847, 0.0507), "R@5": (0.2282, 0.0620), "MRR": (0.1616, 0.0407), "image_R@5": (0.1706, 0.0419), "brain_R@5": (0.1823, 0.0480)}),
            row(heldout_source, "Flat ReGraph+CLIP", 24, {"AUROC": (0.8063, 0.0521), "AUPRC": (0.7869, 0.0460), "R@5": (0.2385, 0.0689), "MRR": (0.1672, 0.0426), "image_R@5": (0.1779, 0.0496), "brain_R@5": (0.1924, 0.0439)}),
            row(heldout_source, "Gated ReGraph/BNT+CLIP", 24, {"AUROC": (0.8104, 0.0574), "AUPRC": (0.7893, 0.0532), "R@5": (0.2466, 0.0717), "MRR": (0.1718, 0.0451), "image_R@5": (0.1874, 0.0429), "brain_R@5": (0.2014, 0.0462)}),
            row(heldout_source, "Gated random embedding", 24, {"AUROC": (0.8128, 0.0567), "AUPRC": (0.7902, 0.0538), "R@5": (0.2489, 0.0649), "MRR": (0.1728, 0.0436), "image_R@5": (0.0407, 0.0125), "brain_R@5": (0.0394, 0.0088)}),
        ],
    )

    hardneg_source = f"{tex}: Table tab:hardneg"
    write_csv(
        final / "table_hard_negative_allfold.csv",
        [
            row(hardneg_source, "ROI-MLP+CLIP", 24, {"AUROC": (0.7651, 0.0386), "AUPRC": (0.7351, 0.0333), "R@5": (0.0831, 0.0273), "MRR": (0.0661, 0.0186), "image_R@5": (0.0772, None), "brain_R@5": (0.0934, None), "brain_MRR": (0.0735, None)}),
            row(hardneg_source, "Flat ReGraph+CLIP", 24, {"AUROC": (0.7658, 0.0329), "AUPRC": (0.7381, 0.0285), "R@5": (0.0879, 0.0298), "MRR": (0.0684, 0.0199), "image_R@5": (0.0804, None), "brain_R@5": (0.0966, None), "brain_MRR": (0.0746, None)}),
            row(hardneg_source, "Gated ReGraph/BNT+CLIP", 24, {"AUROC": (0.7715, 0.0382), "AUPRC": (0.7461, 0.0349), "R@5": (0.0933, 0.0359), "MRR": (0.0712, 0.0236), "image_R@5": (0.0868, None), "brain_R@5": (0.1018, None), "brain_MRR": (0.0786, None)}),
        ],
    )


def write_component_and_ablation_tables(final: Path, tex: str) -> None:
    component_source = f"{tex}: Table tab:sota_baselines"
    write_csv(
        final / "table_phase2_sota_graph_baselines.csv",
        [
            row(component_source, "Our Gated ReGraph+CLIP", 24, {"AUROC": (0.8259, 0.0413), "AUPRC": (0.8065, 0.0460), "R@5": (0.0899, 0.0356), "MRR": (0.0695, 0.0240), "image_R@5": (0.0847, 0.0357), "brain_R@5": (0.0996, 0.0369)}),
            row(component_source, "MindEye2-style shared ROI mapper", 24, {"AUROC": (0.4982, 0.0138), "AUPRC": (0.5006, 0.0117), "R@5": (0.0078, 0.0019), "MRR": (0.0111, 0.0021), "image_R@5": (0.0080, 0.0019), "brain_R@5": (0.0086, 0.0035)}),
            row(component_source, "UMBRAE-style subject encoder", 24, {"AUROC": (0.7415, 0.0678), "AUPRC": (0.7061, 0.0645), "R@5": (0.0308, 0.0105), "MRR": (0.0307, 0.0086), "image_R@5": (0.0329, 0.0131), "brain_R@5": (0.0360, 0.0161)}),
            row(component_source, "MindLink-style subject-adversarial ROI-MLP", 24, {"AUROC": (0.8135, 0.0555), "AUPRC": (0.7873, 0.0575), "R@5": (0.0774, 0.0306), "MRR": (0.0616, 0.0206), "image_R@5": (0.0713, 0.0290), "brain_R@5": (0.0835, 0.0335)}),
            row(component_source, "MindLink-style subject-adversarial ReGraph", 8, {"AUROC": (0.5547, 0.1070), "AUPRC": (0.5475, 0.0957), "R@5": (0.0146, 0.0122), "MRR": (0.0165, 0.0104), "image_R@5": (0.0135, 0.0155), "brain_R@5": (0.0161, 0.0221)}),
        ],
    )

    def simple_rows(source: str, items: list[tuple[str, int, dict[str, tuple[float, float | None]]]]) -> list[dict[str, object]]:
        return [row(source, name, n, values) for name, n, values in items]

    write_csv(
        final / "table_roi_token_controls.csv",
        simple_rows(
            f"{tex}: Table tab:roi_token_controls",
            [
                ("No-adj gated ROI-T", 24, {"AUROC": (0.8258, 0.0513), "AUPRC": (0.8061, 0.0504), "R@5": (0.0884, 0.0361), "MRR": (0.0692, 0.0239), "brain_R@5": (0.0965, 0.0320)}),
                ("Zero ROI embedding", 24, {"AUROC": (0.8258, 0.0513), "AUPRC": (0.8060, 0.0504), "R@5": (0.0883, 0.0358), "MRR": (0.0691, 0.0238), "brain_R@5": (0.0965, 0.0320)}),
                ("Uniform gate", 24, {"AUROC": (0.7549, 0.0472), "AUPRC": (0.7303, 0.0441), "R@5": (0.0653, 0.0242), "MRR": (0.0531, 0.0170), "brain_R@5": (0.0792, 0.0235)}),
                ("Random fixed gate", 24, {"AUROC": (0.7261, 0.0524), "AUPRC": (0.7020, 0.0472), "R@5": (0.0576, 0.0195), "MRR": (0.0478, 0.0131), "brain_R@5": (0.0571, 0.0186)}),
                ("ROI-order shuffle", 24, {"AUROC": (0.5795, 0.0458), "AUPRC": (0.5688, 0.0387), "R@5": (0.0206, 0.0055), "MRR": (0.0212, 0.0044), "brain_R@5": (0.0083, 0.0035)}),
            ],
        ),
    )
    write_csv(
        final / "table_adjacency_perturbation.csv",
        simple_rows(
            f"{tex}: Table tab:adjacency_perturbation",
            [
                ("Default top-k", 1, {"AUROC": (0.5386, 0.0611), "AUPRC": (0.5287, 0.0495), "R@5": (0.0127, 0.0077), "MRR": (0.0153, 0.0062), "brain_R@5": (0.0103, 0.0049)}),
                ("No adjacency", 1, {"AUROC": (0.5337, 0.0416), "AUPRC": (0.5251, 0.0323), "R@5": (0.0129, 0.0067), "MRR": (0.0153, 0.0049), "brain_R@5": (0.0099, 0.0056)}),
                ("Identity", 1, {"AUROC": (0.5340, 0.0601), "AUPRC": (0.5271, 0.0505), "R@5": (0.0128, 0.0104), "MRR": (0.0159, 0.0084), "brain_R@5": (0.0102, 0.0051)}),
                ("Random", 1, {"AUROC": (0.5430, 0.0734), "AUPRC": (0.5332, 0.0604), "R@5": (0.0138, 0.0078), "MRR": (0.0160, 0.0072), "brain_R@5": (0.0099, 0.0053)}),
                ("Drop 50%", 1, {"AUROC": (0.5392, 0.0632), "AUPRC": (0.5301, 0.0492), "R@5": (0.0146, 0.0078), "MRR": (0.0159, 0.0060), "brain_R@5": (0.0098, 0.0053)}),
            ],
        ),
    )
    write_csv(
        final / "table_edge_bias_followup.csv",
        simple_rows(
            f"{tex}: Table tab:edge_bias_followup",
            [
                ("Main all-fold edge-bias", 24, {"AUROC": (0.8219, 0.0509), "AUPRC": (0.8017, 0.0498), "R@5": (0.0844, 0.0311), "brain_R@5": (0.1013, 0.0324)}),
                ("Hard-negative edge-bias", 8, {"AUROC": (0.7613, 0.0419), "AUPRC": (0.7365, 0.0397), "R@5": (0.0812, 0.0315), "brain_R@5": (0.1033, 0.0303)}),
                ("Held-out-image edge-bias", 6, {"AUROC": (0.8418, 0.0130), "AUPRC": (0.8149, 0.0144), "R@5": (0.2760, 0.0963), "brain_R@5": (0.2139, 0.0489)}),
            ],
        ),
    )


def write_qc_and_single_ref(final: Path, tex: str) -> None:
    write_csv(
        final / "split_accounting.csv",
        [
            {"fold": "fold_01", "test_subject": "subj01", "val_subject": "subj08", "train_seq": 3975, "val_seq": 515, "test_seq": 766, "train_pairs": 23850, "val_pairs": 3090, "test_pairs": 4596, "test_imgs": 766, "source": f"{tex}: Table tab:split_accounting"},
            {"fold": "fold_02", "test_subject": "subj02", "val_subject": "subj08", "train_seq": 3975, "val_seq": 515, "test_seq": 766, "train_pairs": 23850, "val_pairs": 3090, "test_pairs": 4596, "test_imgs": 766, "source": f"{tex}: Table tab:split_accounting"},
            {"fold": "fold_03", "test_subject": "subj03", "val_subject": "subj08", "train_seq": 4160, "val_seq": 515, "test_seq": 581, "train_pairs": 24960, "val_pairs": 3090, "test_pairs": 3486, "test_imgs": 581, "source": f"{tex}: Table tab:split_accounting"},
            {"fold": "fold_04", "test_subject": "subj04", "val_subject": "subj08", "train_seq": 4226, "val_seq": 515, "test_seq": 515, "train_pairs": 25356, "val_pairs": 3090, "test_pairs": 3090, "test_imgs": 515, "source": f"{tex}: Table tab:split_accounting"},
            {"fold": "fold_05", "test_subject": "subj05", "val_subject": "subj08", "train_seq": 3975, "val_seq": 515, "test_seq": 766, "train_pairs": 23850, "val_pairs": 3090, "test_pairs": 4596, "test_imgs": 766, "source": f"{tex}: Table tab:split_accounting"},
            {"fold": "fold_06", "test_subject": "subj06", "val_subject": "subj08", "train_seq": 4160, "val_seq": 515, "test_seq": 581, "train_pairs": 24960, "val_pairs": 3090, "test_pairs": 3486, "test_imgs": 581, "source": f"{tex}: Table tab:split_accounting"},
            {"fold": "fold_07", "test_subject": "subj07", "val_subject": "subj08", "train_seq": 3975, "val_seq": 515, "test_seq": 766, "train_pairs": 23850, "val_pairs": 3090, "test_pairs": 4596, "test_imgs": 766, "source": f"{tex}: Table tab:split_accounting"},
            {"fold": "fold_08", "test_subject": "subj08", "val_subject": "subj07", "train_seq": 3975, "val_seq": 766, "test_seq": 515, "train_pairs": 23850, "val_pairs": 4596, "test_pairs": 3090, "test_imgs": 515, "source": f"{tex}: Table tab:split_accounting"},
        ],
    )
    write_csv(
        final / "session_order_pair_qc.csv",
        [
            {"split": "Train", "pairs": 194526, "positive": 97263, "negative": 97263, "complete_groups": 97263, "problem_groups": 0, "anchor_match": "100%", "source": f"{tex}: Table tab:session_order_pair_qc"},
            {"split": "Val", "pairs": 26226, "positive": 13113, "negative": 13113, "complete_groups": 13113, "problem_groups": 0, "anchor_match": "100%", "source": f"{tex}: Table tab:session_order_pair_qc"},
            {"split": "Test", "pairs": 31536, "positive": 15768, "negative": 15768, "complete_groups": 15768, "problem_groups": 0, "anchor_match": "100%", "source": f"{tex}: Table tab:session_order_pair_qc"},
            {"split": "All", "pairs": 252288, "positive": 126144, "negative": 126144, "complete_groups": 126144, "problem_groups": 0, "anchor_match": "100%", "source": f"{tex}: Table tab:session_order_pair_qc"},
        ],
    )
    write_csv(
        final / "fold_difficulty_qc.csv",
        [
            {"fold": "fold_01", "test_subject": "subj01", "test_seq": 766, "repeat_corr": 0.9107, "raw_AUROC": 0.6615, "raw_gap": 0.0151, "model_AUROC": 0.8387, "brain_R@5": 0.0963, "source": f"{tex}: Table tab:fold_difficulty"},
            {"fold": "fold_02", "test_subject": "subj02", "test_seq": 766, "repeat_corr": 0.9057, "raw_AUROC": 0.6697, "raw_gap": 0.0180, "model_AUROC": 0.8440, "brain_R@5": 0.0865, "source": f"{tex}: Table tab:fold_difficulty"},
            {"fold": "fold_03", "test_subject": "subj03", "test_seq": 581, "repeat_corr": 0.8898, "raw_AUROC": 0.6469, "raw_gap": 0.0166, "model_AUROC": 0.8488, "brain_R@5": 0.1279, "source": f"{tex}: Table tab:fold_difficulty"},
            {"fold": "fold_04", "test_subject": "subj04", "test_seq": 515, "repeat_corr": 0.8832, "raw_AUROC": 0.6919, "raw_gap": 0.0188, "model_AUROC": 0.8710, "brain_R@5": 0.1439, "source": f"{tex}: Table tab:fold_difficulty"},
            {"fold": "fold_05", "test_subject": "subj05", "test_seq": 766, "repeat_corr": 0.8997, "raw_AUROC": 0.6824, "raw_gap": 0.0193, "model_AUROC": 0.8783, "brain_R@5": 0.1113, "source": f"{tex}: Table tab:fold_difficulty"},
            {"fold": "fold_06", "test_subject": "subj06", "test_seq": 581, "repeat_corr": 0.8779, "raw_AUROC": 0.6486, "raw_gap": 0.0169, "model_AUROC": 0.8306, "brain_R@5": 0.1038, "source": f"{tex}: Table tab:fold_difficulty"},
            {"fold": "fold_07", "test_subject": "subj07", "test_seq": 766, "repeat_corr": 0.8690, "raw_AUROC": 0.6296, "raw_gap": 0.0144, "model_AUROC": 0.7111, "brain_R@5": 0.0374, "source": f"{tex}: Table tab:fold_difficulty"},
            {"fold": "fold_08", "test_subject": "subj08", "test_seq": 515, "repeat_corr": 0.8524, "raw_AUROC": 0.6158, "raw_gap": 0.0180, "model_AUROC": 0.7848, "brain_R@5": 0.0893, "source": f"{tex}: Table tab:fold_difficulty"},
        ],
    )
    write_csv(
        final / "single_ref_matched_summary.csv",
        [
            row(f"{tex}: Table tab:single_ref_matched", "ROI-MLP+CLIP", 24, {"AUROC": (0.7731, 0.0460), "AUPRC": (0.7450, 0.0472), "R@5": (0.0535, 0.0196), "MRR": (0.0459, 0.0138), "image_R@5": (0.0729, 0.0280), "brain_R@5": (0.0837, 0.0291), "brain_MRR": (0.0668, 0.0193)}),
            row(f"{tex}: Table tab:single_ref_matched", "No-adj gated ROI Transformer+CLIP", 24, {"AUROC": (0.7814, 0.0425), "AUPRC": (0.7584, 0.0435), "R@5": (0.0586, 0.0197), "MRR": (0.0494, 0.0139), "image_R@5": (0.0815, 0.0319), "brain_R@5": (0.0965, 0.0320), "brain_MRR": (0.0756, 0.0208)}),
            row(f"{tex}: Table tab:single_ref_matched", "Gated ReGraph/BNT+CLIP", 24, {"AUROC": (0.7828, 0.0452), "AUPRC": (0.7606, 0.0461), "R@5": (0.0577, 0.0191), "MRR": (0.0497, 0.0138), "image_R@5": (0.0847, 0.0318), "brain_R@5": (0.0996, 0.0310), "brain_MRR": (0.0773, 0.0207)}),
        ],
    )


def write_additional_publication_tables(final: Path, tex: str) -> None:
    write_csv(
        final / "table_within_subject.csv",
        [
            {"model": "Raw Pearson flat", "AUROC": 0.6182, "AUPRC": "", "R@5": 0.0522, "MRR": 0.0433, "source": f"{tex}: Table tab:within_subject"},
            {"model": "ROI MLP + BCE", "AUROC": 0.7672, "AUPRC": 0.7336, "R@5": 0.0543, "MRR": 0.0442, "source": f"{tex}: Table tab:within_subject"},
            {"model": "ROI MLP + BCE+InfoNCE", "AUROC": 0.7778, "AUPRC": 0.7540, "R@5": 0.0758, "MRR": 0.0592, "source": f"{tex}: Table tab:within_subject"},
            {"model": "Naive GCN", "AUROC": 0.5000, "AUPRC": "", "R@5": "", "MRR": "", "source": f"{tex}: Table tab:within_subject"},
            {"model": "BrainGNN Siamese", "AUROC": 0.5000, "AUPRC": "", "R@5": "", "MRR": "", "source": f"{tex}: Table tab:within_subject"},
            {"model": "BNT-token + BCE+InfoNCE", "AUROC": 0.7693, "AUPRC": 0.7512, "R@5": 0.0718, "MRR": 0.0584, "source": f"{tex}: Table tab:within_subject"},
        ],
    )
    write_csv(
        final / "table_graph_only.csv",
        [
            row(f"{tex}: Table tab:graph_only", "Gated ReGraph", 24, {"AUROC": (0.8192, 0.0502), "AUPRC": (0.7940, 0.0510), "R@5": (0.0802, 0.0322), "MRR": (0.0639, 0.0213), "image_R@5": (0.0075, 0.0028), "brain_R@5": (0.0075, 0.0022)}),
            row(f"{tex}: Table tab:graph_only", "Gated ReGraph + CLIP", 24, {"AUROC": (0.8259, 0.0413), "AUPRC": (0.8065, 0.0460), "R@5": (0.0899, 0.0356), "MRR": (0.0695, 0.0240), "image_R@5": (0.0847, 0.0357), "brain_R@5": (0.0996, 0.0369)}),
        ],
    )
    write_csv(
        final / "table_lowshot_calibration.csv",
        [
            {"row_label": "0 & 24", "calibration_images": 0, "n": 24, "image_R@5_mean": 0.1253, "image_R@5_std": 0.0646, "image_MRR_mean": 0.0950, "image_MRR_std": 0.0449, "source": f"{tex}: Table tab:lowshot"},
            {"row_label": "10 & 24", "calibration_images": 10, "n": 24, "image_R@5_mean": 0.0471, "image_R@5_std": 0.0185, "image_MRR_mean": 0.0422, "image_MRR_std": 0.0133, "source": f"{tex}: Table tab:lowshot"},
            {"row_label": "30 & 24", "calibration_images": 30, "n": 24, "image_R@5_mean": 0.0812, "image_R@5_std": 0.0242, "image_MRR_mean": 0.0679, "image_MRR_std": 0.0163, "source": f"{tex}: Table tab:lowshot"},
            {"row_label": "60 & 24", "calibration_images": 60, "n": 24, "image_R@5_mean": 0.1288, "image_R@5_std": 0.0370, "image_MRR_mean": 0.0957, "image_MRR_std": 0.0241, "source": f"{tex}: Table tab:lowshot"},
            {"row_label": "120 & 24", "calibration_images": 120, "n": 24, "image_R@5_mean": 0.1715, "image_R@5_std": 0.0496, "image_MRR_mean": 0.1242, "image_MRR_std": 0.0316, "source": f"{tex}: Table tab:lowshot"},
        ],
    )
    write_csv(
        final / "table_external_visual_roi_smoke.csv",
        [
            row(f"{tex}: Table tab:external_visual_roi_smoke", "BOLD5000 visual ROI & ROI-MLP", 18, {"AUROC": (0.6561, 0.0315), "AUPRC": (0.0094, 0.0013), "R@5": (0.0539, 0.0143), "MRR": (0.0533, 0.0084)}),
            row(f"{tex}: Table tab:external_visual_roi_smoke", "BOLD5000 visual ROI & Gated ROI Transformer", 18, {"AUROC": (0.6240, 0.0611), "AUPRC": (0.0085, 0.0020), "R@5": (0.0561, 0.0139), "MRR": (0.0513, 0.0103)}),
            row(f"{tex}: Table tab:external_visual_roi_smoke", "CNeuroMod visual ROI & ROI-MLP", 18, {"AUROC": (0.6248, 0.0212), "AUPRC": (0.0164, 0.0023), "R@5": (0.1058, 0.0147), "MRR": (0.0879, 0.0102)}),
            row(f"{tex}: Table tab:external_visual_roi_smoke", "CNeuroMod visual ROI & Gated ROI Transformer", 18, {"AUROC": (0.6071, 0.0423), "AUPRC": (0.0159, 0.0034), "R@5": (0.0979, 0.0184), "MRR": (0.0827, 0.0111)}),
            row(f"{tex}: Table tab:external_visual_roi_smoke", "THINGS-fMRI visual ROI & ROI-MLP", 9, {"AUROC": (0.5777, 0.0352), "AUPRC": (0.0067, 0.0011), "R@5": (0.0506, 0.0159), "MRR": (0.0456, 0.0092)}),
            row(f"{tex}: Table tab:external_visual_roi_smoke", "THINGS-fMRI visual ROI & Gated ROI Transformer", 9, {"AUROC": (0.5291, 0.0377), "AUPRC": (0.0057, 0.0008), "R@5": (0.0308, 0.0132), "MRR": (0.0335, 0.0076)}),
            row(f"{tex}: Table tab:external_visual_roi_smoke", "LAION-fMRI visual ROI & ROI-MLP", 30, {"AUROC": (0.5315, 0.0213), "AUPRC": (0.0284, 0.0022), "R@5": (0.1481, 0.0185), "MRR": (0.1215, 0.0103)}),
            row(f"{tex}: Table tab:external_visual_roi_smoke", "LAION-fMRI visual ROI & Gated ROI Transformer", 30, {"AUROC": (0.5296, 0.0221), "AUPRC": (0.0284, 0.0023), "R@5": (0.1574, 0.0283), "MRR": (0.1252, 0.0126)}),
        ],
    )
    write_csv(
        final / "table_gate_confound.csv",
        [
            {"row_label": "$|\\Delta_{21}|$", "spearman": 0.4184, "partial_spearman": -0.0740, "n_roi": 180, "source": f"{tex}: Table tab:gate_confound"},
            {"row_label": "$|\\Delta_{31}|$", "spearman": 0.4215, "partial_spearman": -0.0653, "n_roi": 180, "source": f"{tex}: Table tab:gate_confound"},
            {"row_label": "$|\\Delta_{32}|$", "spearman": 0.4073, "partial_spearman": -0.1252, "n_roi": 180, "source": f"{tex}: Table tab:gate_confound"},
        ],
    )
    write_csv(
        final / "table_matched_deletion.csv",
        [
            {"row_label": "Property-matched random & 5", "deletion_set": "Property-matched random", "k": 5, "AUROC_drop": 0.0301, "R@5_drop": 0.0105, "brain_R@5_drop": 0.0168, "source": f"{tex}: Table tab:matched_deletion"},
            {"row_label": "Top gated ROIs & 5", "deletion_set": "Top gated ROIs", "k": 5, "AUROC_drop": 0.0548, "R@5_drop": 0.0186, "brain_R@5_drop": 0.0326, "source": f"{tex}: Table tab:matched_deletion"},
            {"row_label": "Property-matched random & 20", "deletion_set": "Property-matched random", "k": 20, "AUROC_drop": 0.0714, "R@5_drop": 0.0218, "brain_R@5_drop": 0.0384, "source": f"{tex}: Table tab:matched_deletion"},
            {"row_label": "Top gated ROIs & 20", "deletion_set": "Top gated ROIs", "k": 20, "AUROC_drop": 0.1106, "R@5_drop": 0.0461, "brain_R@5_drop": 0.0586, "source": f"{tex}: Table tab:matched_deletion"},
            {"row_label": "Property-matched random & 40", "deletion_set": "Property-matched random", "k": 40, "AUROC_drop": 0.0244, "R@5_drop": 0.0094, "brain_R@5_drop": 0.0111, "source": f"{tex}: Table tab:matched_deletion"},
            {"row_label": "Top gated ROIs & 40", "deletion_set": "Top gated ROIs", "k": 40, "AUROC_drop": 0.2577, "R@5_drop": 0.0684, "brain_R@5_drop": 0.0827, "source": f"{tex}: Table tab:matched_deletion"},
        ],
    )


def write_story_and_external(final: Path, external: Path, tex: str) -> None:
    final.mkdir(parents=True, exist_ok=True)
    story = """# AAAI ROI-Token Story Summary

Recommended claim: fixed anatomical ROI-token modeling, gated ROI-preserving readout, and image alignment improve cross-subject natural-image fMRI retrieval.

Do not claim that explicit fixed adjacency is the source of the gain. The no-adjacency gated ROI Transformer is statistically tied with the final BNT/ReGraph ROI-token variant, and static adjacency perturbations plus learned edge-bias follow-ups do not establish a separate fixed-edge contribution.

Core evidence:
- Gated ReGraph/BNT+CLIP and no-adj gated ROI Transformer+CLIP both outperform ROI-MLP+CLIP in the main all-fold setting.
- ROI-order shuffle and gate controls show that fixed ROI-token layout and learned gates are important.
- Single-reference session-matched controls preserve the ROI-token conclusion.
- External visual-ROI smoke checks show above-chance cross-subject signal outside NSD but are not full HCP-MMP external validations.
"""
    (final / "aaai_roi_token_story_summary.md").write_text(story, encoding="utf-8")

    external.mkdir(parents=True, exist_ok=True)
    external_summary = f"""# External Visual-ROI Smoke Summary

Source: {tex}: Table tab:external_visual_roi_smoke.

These checks use public visual-ROI summaries or public beta-map derivatives from CNeuroMod-THINGS, BOLD5000, THINGS-fMRI, and LAION-fMRI. They are not full HCP-MMP 180-ROI external validations.

| Dataset | Model | n | AUROC | AUPRC | R@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BOLD5000 visual ROI | ROI-MLP | 18 | 0.6561 +/- 0.0315 | 0.0094 +/- 0.0013 | 0.0539 +/- 0.0143 | 0.0533 +/- 0.0084 |
| BOLD5000 visual ROI | Gated ROI Transformer | 18 | 0.6240 +/- 0.0611 | 0.0085 +/- 0.0020 | 0.0561 +/- 0.0139 | 0.0513 +/- 0.0103 |
| CNeuroMod visual ROI | ROI-MLP | 18 | 0.6248 +/- 0.0212 | 0.0164 +/- 0.0023 | 0.1058 +/- 0.0147 | 0.0879 +/- 0.0102 |
| CNeuroMod visual ROI | Gated ROI Transformer | 18 | 0.6071 +/- 0.0423 | 0.0159 +/- 0.0034 | 0.0979 +/- 0.0184 | 0.0827 +/- 0.0111 |
| THINGS-fMRI visual ROI | ROI-MLP | 9 | 0.5777 +/- 0.0352 | 0.0067 +/- 0.0011 | 0.0506 +/- 0.0159 | 0.0456 +/- 0.0092 |
| THINGS-fMRI visual ROI | Gated ROI Transformer | 9 | 0.5291 +/- 0.0377 | 0.0057 +/- 0.0008 | 0.0308 +/- 0.0132 | 0.0335 +/- 0.0076 |
| LAION-fMRI visual ROI | ROI-MLP | 30 | 0.5315 +/- 0.0213 | 0.0284 +/- 0.0022 | 0.1481 +/- 0.0185 | 0.1215 +/- 0.0103 |
| LAION-fMRI visual ROI | Gated ROI Transformer | 30 | 0.5296 +/- 0.0221 | 0.0284 +/- 0.0023 | 0.1574 +/- 0.0283 | 0.1252 +/- 0.0126 |

Interpretation: these datasets support above-chance cross-subject same-image signal outside NSD, but they do not reproduce the main NSD model ordering. They should be presented as external feasibility checks, not as full external validation of the HCP-MMP ReGraph-VLM result.
"""
    (external / "external_visual_roi_all4_summary.md").write_text(external_summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_main_tables(args.final_tables_dir, args.source_tex)
    write_component_and_ablation_tables(args.final_tables_dir, args.source_tex)
    write_qc_and_single_ref(args.final_tables_dir, args.source_tex)
    write_additional_publication_tables(args.final_tables_dir, args.source_tex)
    write_story_and_external(args.final_tables_dir, args.external_summary_dir, args.source_tex)
    print(f"Wrote publication artifacts to {args.final_tables_dir} and {args.external_summary_dir}")


if __name__ == "__main__":
    main()

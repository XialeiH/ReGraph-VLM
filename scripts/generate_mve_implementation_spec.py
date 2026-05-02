from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PDF = ROOT / "mve_implementation_spec_v1.pdf"
MIRROR_PDF = ROOT / "output" / "pdf" / "mve_implementation_spec_v1.pdf"


PALETTE = {
    "ink": colors.HexColor("#1F2933"),
    "muted": colors.HexColor("#52606D"),
    "accent": colors.HexColor("#0F4C81"),
    "accent_light": colors.HexColor("#EAF2FA"),
    "grid": colors.HexColor("#D9E2EC"),
    "soft": colors.HexColor("#F7F9FC"),
    "success": colors.HexColor("#1F7A4C"),
}


def styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            name="TitleSpec",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=PALETTE["ink"],
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            name="SubTitleSpec",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=PALETTE["muted"],
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    base.add(
        ParagraphStyle(
            name="SectionSpec",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=PALETTE["accent"],
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    base.add(
        ParagraphStyle(
            name="BodySpec",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=11.5,
            textColor=PALETTE["ink"],
            alignment=TA_LEFT,
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="SmallSpec",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10,
            textColor=PALETTE["muted"],
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="BoxSpec",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.7,
            leading=10.8,
            textColor=PALETTE["ink"],
            spaceAfter=0,
        )
    )
    base.add(
        ParagraphStyle(
            name="FormulaSpec",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.2,
            leading=10.2,
            textColor=PALETTE["ink"],
            leftIndent=6,
            spaceBefore=2,
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="TableHeaderSpec",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.1,
            leading=10.1,
            textColor=colors.white,
        )
    )
    base.add(
        ParagraphStyle(
            name="TableBodySpec",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.0,
            leading=10.0,
            textColor=PALETTE["ink"],
        )
    )
    return base


def p(text, style_name, st):
    return Paragraph(text, st[style_name])


def make_table(data, col_widths, st, header_rows=1, body_font=8.2):
    wrapped = []
    for row_idx, row in enumerate(data):
        style = st["TableHeaderSpec"] if row_idx < header_rows else st["TableBodySpec"]
        wrapped.append([Paragraph(str(cell), style) for cell in row])
    table = Table(wrapped, colWidths=col_widths, repeatRows=header_rows, hAlign="LEFT")
    style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), PALETTE["accent"]),
            ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
            ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), body_font),
            ("LEADING", (0, 0), (-1, -1), body_font + 2),
            ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, PALETTE["soft"]]),
            ("GRID", (0, 0), (-1, -1), 0.4, PALETTE["grid"]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )
    table.setStyle(style)
    return table


def info_box(title, body, doc_width, st):
    table = Table(
        [[p(title, "BoxSpec", st)], [p(body, "BodySpec", st)]],
        colWidths=[doc_width],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALETTE["accent_light"]),
                ("BOX", (0, 0), (-1, -1), 0.6, PALETTE["accent"]),
                ("INNERGRID", (0, 0), (-1, -1), 0.0, colors.white),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(PALETTE["grid"])
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, A4[1] - 16 * mm, A4[0] - doc.rightMargin, A4[1] - 16 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(PALETTE["muted"])
    canvas.drawString(doc.leftMargin, A4[1] - 13 * mm, "MVE Implementation Spec v1")
    canvas.drawRightString(
        A4[0] - doc.rightMargin,
        10 * mm,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def build_story(doc, st):
    story = []
    width = doc.width

    story.append(Spacer(1, 10 * mm))
    story.append(p("MVE Implementation Spec v1", "TitleSpec", st))
    story.append(
        p(
            "Version 0 technical specification for experiment-ready cross-subject shared perceptual cell learning on NSD",
            "SubTitleSpec",
            st,
        )
    )
    story.append(
        info_box(
            "Version 0 objective",
            "Lock the first experiment to one falsifiable claim: a learned shared cell space should improve held-out-subject closed-set stimulus identification on NSD relative to strong non-graph baselines. Version 0 intentionally freezes the scope before graph reasoning.",
            width,
            st,
        )
    )
    story.append(Spacer(1, 5))
    story.append(
        p(
            "<b>Version boundary.</b> This document replaces proposal-level ambiguity with coding-level defaults. It assumes NSD only, the shared 1000-image stimulus subset, averaged repeated-image responses, no dynamic graph module, and a single primary metric for the first pass.",
            "BodySpec",
            st,
        )
    )

    story.append(p("1. Sample Definition", "SectionSpec", st))
    sample_table = [
        ["Item", "Version 0 decision"],
        ["Dataset", "NSD only for primary development and acceptance testing."],
        [
            "Sample unit",
            "One subject x one shared image, using the subject's averaged repeated-image beta response for that image.",
        ],
        [
            "Label",
            "Closed-set image identity over the 1000 NSD shared stimuli. This keeps the first metric simple and directly tied to cross-subject alignment.",
        ],
        [
            "Split",
            "Outer loop = 8 leave-one-subject-out folds. Per fold: 1 held-out test subject, 1 validation subject, 6 training subjects.",
        ],
        [
            "Train/val/test policy",
            "No stimulus split in Version 0. All 1000 shared images appear in every split; only the subject axis changes. This isolates subject transfer rather than open-set semantic generalization.",
        ],
    ]
    story.append(make_table(sample_table, [50 * mm, width - 50 * mm], st))
    story.append(
        p(
            "<b>Rationale.</b> Averaging repeated presentations reduces trial noise and lets the first implementation answer the alignment question before single-trial variance, temporal modeling, or external datasets are introduced.",
            "BodySpec",
            st,
        )
    )

    story.append(p("2. Input Representation", "SectionSpec", st))
    input_table = [
        ["Component", "Specification"],
        ["Signal level", "Voxel-level beta pattern restricted to selected visual ROIs; no ROI mean pooling for the main model."],
        ["ROIs", "NSD visual ROI union: V1, V2, V3, hV4, LO, VO, and PHC masks from official NSD ROI annotations."],
        ["Raw tensor", "Subject-specific vector with all voxels in the selected ROI union before projection."],
        ["Normalization", "Fit voxelwise z-score statistics on training subjects only; apply the fitted mean/std to val and test subjects per fold."],
        ["Projection", "Subject-wise PCA fitted on training samples only, projected to d_in = 512. No whitening in Version 0."],
        ["Model input shape", "[512] per sample after projection; batch shape [B, 512]."],
    ]
    story.append(make_table(input_table, [42 * mm, width - 42 * mm], st))
    story.append(
        p(
            "<b>Stored artifacts.</b> Keep both the raw ROI voxel indices and the fitted PCA objects so that later diagnostics can map prototypes back to anatomy without rerunning extraction.",
            "BodySpec",
            st,
        )
    )
    story.append(PageBreak())

    story.append(p("3. Version 0 Model Definition", "SectionSpec", st))
    story.append(
        p(
            "<b>Architecture.</b> The proposed model is static and sequence-free in Version 0: subject encoder -> shared prototype bank -> cell activation vector -> classifier head. Dynamic GNN blocks are explicitly deferred.",
            "BodySpec",
            st,
        )
    )
    model_table = [
        ["Module", "Default setting"],
        ["Subject encoder f_s", "Subject-specific two-layer MLP: 512 -> 256 -> 256 with GELU, dropout 0.1, and LayerNorm on the output."],
        ["Shared prototype bank P", "P in R^(K x d), with K = 64 shared cells and d = 256 embedding dimensions."],
        ["Assignment rule", "Cosine similarity followed by softmax with temperature tau = 0.07. No Sinkhorn and no hard top-k masking in Version 0."],
        ["Cell activation z", "K-dimensional activation vector z = a. The model uses one activation vector per sample, not a T x K sequence."],
        ["Output head g", "LayerNorm -> Linear(64, 256) -> GELU -> Linear(256, 1000)."],
    ]
    story.append(make_table(model_table, [42 * mm, width - 42 * mm], st))
    story.append(
        Preformatted(
            "x_s in R^512\nh = f_s(x_s) in R^256\nu_k = cos(h, p_k)\na = softmax(u / tau),  tau = 0.07\nz = a in R^64\ny_hat = g(z)\nL = L_ce + 0.20 L_align + 0.05 L_balance + 0.01 L_orth",
            st["FormulaSpec"],
        )
    )
    story.append(
        p(
            "<b>L_align.</b> For each image represented by multiple training subjects in a batch, minimize 1 - cosine(z_i, z_j) across subject pairs. "
            "<b>L_balance.</b> Match the batch-mean assignment distribution to Uniform(64) with KL divergence. "
            "<b>L_orth.</b> Penalize prototype redundancy with ||PP^T - I||<super>2</super><sub>F</sub>.",
            "BodySpec",
            st,
        )
    )

    story.append(p("4. Baseline Hierarchy", "SectionSpec", st))
    baseline_table = [
        ["ID", "Baseline", "Purpose"],
        ["B1", "ROI feature + linear classifier", "Lowest-friction atlas-aware baseline on the same 512-d projected input."],
        ["B2", "ROI feature + two-layer MLP", "Nonlinear baseline without shared cell structure."],
        ["B3", "Shared linear latent space", "Subject-specific linear adapters to 256-d shared latent plus linear classifier."],
        ["B4", "Shared encoder without prototype bank", "Strong non-graph baseline: same subject encoder as the proposed model, but h goes directly to the classifier head."],
        ["B5", "Prototype bank model", "Proposed Version 0 model and the only new method required for acceptance."],
    ]
    story.append(make_table(baseline_table, [14 * mm, 58 * mm, width - 72 * mm], st))
    story.append(
        p(
            "<b>Decision rule.</b> Graph modules will not be introduced unless B5 beats both B3 and B4 on held-out subjects. Beating only weak ROI baselines is not sufficient.",
            "BodySpec",
            st,
        )
    )

    story.append(p("5. Training Protocol", "SectionSpec", st))
    train_table = [
        ["Item", "Version 0 default"],
        ["Batch construction", "Sample 32 image IDs per step and draw up to 4 training subjects per image; target batch size is 128."],
        ["Subject mixing", "Yes. Every batch mixes subjects so that L_align is defined within the batch."],
        ["Optimizer", "AdamW with parameter groups."],
        ["Learning rates", "1e-3 for subject encoders and classifier head; 5e-4 for prototype bank."],
        ["Regularization", "Weight decay 1e-4, dropout 0.1, gradient clipping at global norm 1.0."],
        ["Schedule", "5 warmup epochs then cosine decay to 1e-5."],
        ["Epochs", "80 max epochs."],
        ["Early stopping", "Patience 10 on validation top-1 accuracy; break ties with validation top-5 accuracy."],
        ["Seeds", "5 fixed seeds: 11, 22, 33, 44, 55."],
    ]
    story.append(make_table(train_table, [40 * mm, width - 40 * mm], st))
    story.append(PageBreak())

    story.append(p("6. Evaluation Protocol", "SectionSpec", st))
    eval_table = [
        ["Level", "Metric or report"],
        ["Primary", "Held-out-subject top-1 closed-set accuracy over the 1000 shared images."],
        ["Secondary", "Top-5 accuracy, retrieval Recall@1 and Recall@5 from the penultimate embedding, and RSA between subject-level representation dissimilarity matrices."],
        ["Consistency analysis", "Cell consistency = mean same-image cross-subject cosine on z plus top-1 prototype agreement rate."],
        ["Reporting", "Mean +/- std across 8 subject folds and 5 seeds, with per-subject tables retained in the results directory."],
        ["Acceptance gate", "Prototype model should exceed B4 by at least 3 absolute points on mean top-1 accuracy or win on at least 6 of 8 held-out subjects."],
    ]
    story.append(make_table(eval_table, [36 * mm, width - 36 * mm], st))
    story.append(
        p(
            "<b>External validation.</b> THINGS-fMRI and BOLD5000 remain frozen-transfer checks after NSD succeeds. They are not part of the Version 0 acceptance gate, but the evaluation scripts should already support loading an external feature matrix and running a linear probe on frozen z.",
            "BodySpec",
            st,
        )
    )

    story.append(p("7. Failure Diagnostics", "SectionSpec", st))
    diag_table = [
        ["Failure mode", "Detection signal", "First response"],
        ["Prototype collapse", "exp(H(mean(a))) < 16 or one cell gets > 25% of assignments.", "Raise L_balance, reduce K to 32, inspect assignment histograms."],
        ["Cross-subject misalignment", "Same-image cosine on z is not at least 0.10 above different-image cosine.", "Audit z-scoring, PCA fitting, and subject-mixed batches first."],
        ["Overfitting", "Train top-1 is > 10 points above validation for 5 straight epochs.", "Increase dropout, shrink hidden width, and stop adding objectives."],
        ["Weak external transfer", "NSD wins but frozen transfer to THINGS/BOLD5000 is unstable.", "Check ROI overlap, preprocessing leakage, and label mapping before claiming dataset shift."],
    ]
    story.append(make_table(diag_table, [30 * mm, 48 * mm, width - 78 * mm], st, body_font=7.7))
    story.append(PageBreak())

    story.append(p("8. First Coding Milestones", "SectionSpec", st))
    milestone_table = [
        ["Module", "Output artifact"],
        ["Data manifest builder", "data/nsd_v0/shared1000_manifest.csv and subject-fold split JSON files."],
        ["ROI extraction and PCA", "data/nsd_v0/features/subjXX_visual_pca512.npz and artifacts/nsd_v0/preprocess/subjXX_pca.pkl."],
        ["Baseline training script", "results/nsd_v0/baselines/{model}/fold_{k}_seed_{s}/metrics.json and checkpoint.pt."],
        ["Prototype model trainer", "results/nsd_v0/prototype/fold_{k}_seed_{s}/model.pt, metrics.json, assignment_stats.json."],
        ["Evaluation script", "results/nsd_v0/eval/summary.csv and subject_breakdown.csv."],
        ["Visualization script", "results/nsd_v0/figures/prototype_usage.png, alignment_heatmap.png, subject_rsa.png."],
    ]
    story.append(make_table(milestone_table, [44 * mm, width - 44 * mm], st))
    story.append(
        info_box(
            "Implementation order",
            "Week 1 should end with the NSD shared-stimulus manifest and projected features on disk. Week 2 should end with B1-B4 training runs and a stable evaluation script. The prototype model should start only after those artifacts exist.",
            width,
            st,
        )
    )
    story.append(Spacer(1, 5))
    story.append(
        p(
            "<b>Repository-ready checklist.</b> Every milestone above should map to one config file, one executable script, and one machine-readable output. If a milestone only produces narrative notes, it is not complete.",
            "BodySpec",
            st,
        )
    )
    story.append(
        p(
            "<b>Deferred items.</b> Single-trial beta modeling, temporal bins, graph edges between cells, and external-dataset fine-tuning are explicitly outside Version 0. If any of these appear in the first implementation PR, the scope has drifted.",
            "BodySpec",
            st,
        )
    )
    story.append(
        p(
            "This specification is experiment-ready when every table entry above can be mapped to a config key, a saved artifact, or a unit-tested dataloader assumption.",
            "SmallSpec",
            st,
        )
    )

    return story


def build_pdf(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    st = styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=20 * mm,
        bottomMargin=16 * mm,
        pageCompression=0,
        title="MVE Implementation Spec v1",
        author="OpenAI Codex",
    )
    doc.build(build_story(doc, st), onFirstPage=header_footer, onLaterPages=header_footer)


def main():
    build_pdf(OUTPUT_PDF)
    build_pdf(MIRROR_PDF)
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {MIRROR_PDF}")


if __name__ == "__main__":
    main()

# External Validation Dataset Scan

This note tracks external fMRI datasets for reviewer-facing validation. Large datasets should be downloaded directly to Shanghai HPC scratch under:

```text
/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation
```

## Priority Order

1. **LAION-fMRI**

   Strong future external-validation target. The launch release describes five participants, 25,052 unique natural images, 1,492 shared images, repeated shared images, 7T acquisition, open data access, and GLMsingle single-trial beta estimates. This may be the best next dataset for a paper-level external validation because it has repeated image responses and public single-trial betas. It has not yet been downloaded or tested in this project. If used, all downloads should go directly to Shanghai HPC scratch under `external_validation/laion_fmri`.

2. **CNeuroMod-THINGS**

   Best fit for the current paper. It has four participants, shared THINGS object images, and three presentations per image. This is the closest external replication of the strict repeated-image and cross-subject same-image retrieval setup.

3. **BOLD5000**

   Strong image-retrieval validation target with four subjects and natural images from COCO, SUN, and ImageNet. It is not a strict T=3 repetition replication, but it is useful for external brain-image retrieval and NSD-to-BOLD5000 transfer because BOLD5000 intentionally overlaps with NSD stimuli.

4. **Natural Object Dataset (NOD)**

   Large-scale stress test with many subjects and naturalistic images. It is valuable for external generalization but likely requires substantial preprocessing and ROI extraction work before it can be used in the current HCP-MMP ROI-token pipeline.

5. **THINGS-fMRI**

   Useful for semantic/concept validation with rich THINGS annotations. It is less directly matched to the current repeated-image setup than CNeuroMod-THINGS.

## Source Check Notes

Current public dataset pages checked on 2026-05-30:

```text
LAION-fMRI:
  https://laion-fmri.hebartlab.com/
  Relevant notes: five participants, 25,052 unique images, 1,492 shared launch-release images,
  repeated shared images, 7T acquisition, GLMsingle beta estimates, open research access.
  Data access notes: CC0 fMRI derivatives are in the public AWS S3 bucket `s3://laion-fmri`;
  raw stimulus images require the dataset DUA flow and should not be fetched without that step.

CNeuroMod-THINGS:
  https://www.nature.com/articles/s41597-026-06591-y
  Relevant notes: four participants, 33--36 sessions, 4,320 THINGS images, three image repetitions,
  DataLad/CONP access, trial-wise beta derivatives described.

BOLD5000:
  https://bold5000-dataset.github.io/website/
  Relevant notes: four subjects, 5,254 images, COCO/SUN/ImageNet stimuli, Release 2.0 available.
```

Shanghai HPC access probe:

```text
probe date: 2026-05-30
probe path: /gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/laion_fmri_probe
root listing: https://laion-fmri.s3.amazonaws.com/?list-type=2&max-keys=50
GLMsingle listing: https://laion-fmri.s3.amazonaws.com/?list-type=2&prefix=derivatives/glmsingle-tedana/sub-01/&max-keys=50
status: public S3 HTTPS listing works from Shanghai HPC; `aws` CLI was not installed on the login shell.
large downloads performed: none
```

Public GLMsingle subject inventory from the Shanghai probe:

```text
subject IDs exposed under derivatives/glmsingle-tedana:
  sub-01, sub-03, sub-05, sub-06, sub-07

per subject:
  34 sessions
  34 SingletrialBetas trial TSVs
  34 SingletrialBetas NIfTI beta maps

estimated listed public data size:
  sub-01: 80.250 GB listed, 38.956 GB singletrial beta maps
  sub-03: 85.306 GB listed, 41.370 GB singletrial beta maps
  sub-05: 82.089 GB listed, 39.771 GB singletrial beta maps
  sub-06: 76.200 GB listed, 37.097 GB singletrial beta maps
  sub-07: 78.060 GB listed, 37.955 GB singletrial beta maps

current status:
  metadata/listing feasibility confirmed only
  full validation still requires ROI projection/extraction and stimulus alignment
```

## Immediate Implementation Target

Use CNeuroMod-THINGS first. The first concrete milestone should be a smoke validation fold:

```text
one or two subjects
one small session subset
trial metadata/events
preprocessed derivative or GLM beta subset
ROI-cache construction
same-image repeat retrieval sanity check
```

Only after the smoke fold works should the download expand to all four subjects.

## CNeuroMod-THINGS Progress On Shanghai HPC

All CNeuroMod files below were accessed directly on Shanghai HPC scratch under:

```text
/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/cneuromod_things
```

Metadata/event status:

```text
event TSV files fetched: 834
read failures: 0
valid rows after ran/not_for_memory filters: 46,080
subjects: 4
unique images after filters: 4,320
images with strict T=3 repeats in all four subjects: 2,107
repeat-matched cross-subject positive pairs for all-four-subject strict T=3 images: 37,926
```

Derivative access status:

```text
Publicly available through CONP/DataLad:
  - event TSVs
  - public descriptive visual-ROI beta arrays

Not publicly fetchable from the configured DataLad HTTP remote:
  - MNI152NLin2009cAsym trialBetas HDF5
  - MNI152NLin2009cAsym imageBetas HDF5
```

Because the full MNI trial HDF5 files are only available on private or unavailable siblings in the current DataLad metadata, the first executable smoke test uses the public `descriptive` fLoc visual-ROI beta arrays rather than full HCP-MMP volumetric extraction.

Current external smoke output:

```text
/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/cneuromod_things/visual_roi_scalar4_smoke
```

Smoke setup:

```text
subjects: sub-01 and sub-02
shared strict-T3 images used: 500
trials per subject: 1,500
ROI tokens: EBA, FFA, OFA, OPA, PPA, pSTS
feature format: scalar4, padded to 180 nodes for code compatibility
```

Raw cosine external smoke result:

```text
AUROC: 0.5650
AUPRC: 0.00254
R@5: 0.0160
MRR: 0.01935
chance R@5: 0.0100
chance AUPRC: 0.0020
```

Trained external smoke result, mean +/- std over 3 seeds:

```text
Model                       AUROC            AUPRC            R@5              MRR
ROI-MLP                     0.5903 +/- 0.0254 0.0145 +/- 0.0016 0.0850 +/- 0.0044 0.0773 +/- 0.0043
Gated ROI Transformer       0.5961 +/- 0.0286 0.0147 +/- 0.0024 0.0911 +/- 0.0096 0.0790 +/- 0.0075
```

Interpretation:

```text
This is a metadata and data-pipeline feasibility result, not a final external ReGraph-VLM validation result. It shows that the public CNeuroMod derivatives contain weak but above-chance cross-subject same-image signal in visual fLoc ROI summaries. The gated ROI transformer is slightly stronger than ROI-MLP in this public visual-ROI smoke setting, but the feature space is only six visual fLoc ROIs padded to 180 tokens. A full paper-level external validation still requires public access to full trial-wise MNI/T1w beta derivatives or another dataset with directly usable HCP-MMP/fsLR beta maps.
```

## BOLD5000 Progress On Shanghai HPC

All BOLD5000 files below were accessed directly on Shanghai HPC scratch under:

```text
/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/bold5000
```

Downloaded public smoke files:

```text
Release 2.0 GLMsingle ROI beta zip: 229M
CSI1 session-01 TypeD full beta NIfTI: 268M
CSI2 session-01 TypeD full beta NIfTI: 274M
OpenNeuro ds001499 metadata/events clone
```

Data status:

```text
Full beta NIfTI files are subject-space volumes, not directly HCP-MMP/MNI-aligned.
The ROI beta zip contains public GLMsingle visual ROI arrays for 4 subjects.
OpenNeuro event TSV files provide image order, allowing same-image cross-subject pairing.
```

Visual-ROI scalar4 smoke setup:

```text
subjects used: CSI1, CSI2, CSI3
subject pairs: CSI1-CSI2, CSI1-CSI3, CSI2-CSI3
shared single-presentation images used: 1,000
ROI tokens: bilateral EarlyVis, LOC/LO, OPA, PPA, RSC/RRSC
feature format: scalar4, padded to 180 nodes for code compatibility
models: ROI-MLP and gated ROI Transformer
seeds: 11, 22, 33
```

Trained BOLD5000 external smoke result, mean +/- std over 3 seeds:

```text
Pair       Model                       AUROC            AUPRC            R@5              MRR
CSI1-CSI2  ROI-MLP                     0.6411 +/- 0.0381 0.0081 +/- 0.0009 0.0567 +/- 0.0080 0.0521 +/- 0.0082
CSI1-CSI2  Gated ROI Transformer       0.6144 +/- 0.0420 0.0075 +/- 0.0013 0.0450 +/- 0.0132 0.0473 +/- 0.0111
CSI1-CSI3  ROI-MLP                     0.6615 +/- 0.0215 0.0097 +/- 0.0010 0.0592 +/- 0.0052 0.0544 +/- 0.0042
CSI1-CSI3  Gated ROI Transformer       0.6288 +/- 0.0412 0.0087 +/- 0.0013 0.0675 +/- 0.0025 0.0590 +/- 0.0037
CSI2-CSI3  ROI-MLP                     0.6078 +/- 0.0092 0.0071 +/- 0.0010 0.0342 +/- 0.0118 0.0392 +/- 0.0098
CSI2-CSI3  Gated ROI Transformer       0.5963 +/- 0.0479 0.0072 +/- 0.0013 0.0475 +/- 0.0132 0.0441 +/- 0.0073
```

Interpretation:

```text
BOLD5000 confirms that the visual-ROI scalar4 pipeline can recover above-chance cross-subject same-image signal on an independent natural-image fMRI dataset. However, unlike CNeuroMod, this smoke result favors the simpler ROI-MLP on AUROC, while retrieval metrics are mixed. It should be reported, if used at all, as external feasibility evidence and not as evidence that the gated ROI transformer universally dominates on external visual-ROI summaries.
```

## Four-Subject External Visual-ROI Expansion

The initial external smoke checks were expanded to all available public visual-ROI subjects for CNeuroMod-THINGS and BOLD5000. These results remain public visual-ROI checks rather than full HCP-MMP 180-ROI validation. They should be used as reviewer-facing feasibility evidence and as guidance for future external validation, not as a replacement for a full external atlas-ROI replication.

Summary outputs are stored on Shanghai HPC:

```text
/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/summary/external_visual_roi_all4_all_runs.csv
/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/summary/external_visual_roi_all4_mean_std_by_pair_model.csv
/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/summary/external_visual_roi_all4_mean_std_by_dataset_model.csv
/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/summary/external_visual_roi_all4_summary.md
```

Four-subject CNeuroMod setup:

```text
subjects: sub-01, sub-02, sub-03, sub-06
subject pairs: 6
shared strict-T3 images used: 500
trials per subject: 1,500
ROI tokens: EBA, FFA, OFA, OPA, PPA, pSTS
feature format: scalar4, padded to 180 nodes for code compatibility
models: ROI-MLP and gated ROI Transformer
seeds: 11, 22, 33
```

Four-subject BOLD5000 setup:

```text
subjects: CSI1, CSI2, CSI3, CSI4
subject pairs: 6
shared single-presentation images used: 1,000
ROI tokens: bilateral EarlyVis, LOC/LO, OPA, PPA, RSC/RRSC
feature format: scalar4, padded to 180 nodes for code compatibility
models: ROI-MLP and gated ROI Transformer
seeds: 11, 22, 33
```

Overall all-four-subject results, mean +/- std across subject-pair x seed runs:

```text
Dataset                     Model                       AUROC             AUPRC             R@5               MRR
BOLD5000 visual ROI all4     ROI-MLP                     0.6561 +/- 0.0315 0.0094 +/- 0.0013 0.0539 +/- 0.0143 0.0533 +/- 0.0084
BOLD5000 visual ROI all4     Gated ROI Transformer       0.6240 +/- 0.0611 0.0085 +/- 0.0020 0.0561 +/- 0.0139 0.0513 +/- 0.0103
CNeuroMod visual ROI all4    ROI-MLP                     0.6248 +/- 0.0212 0.0164 +/- 0.0023 0.1058 +/- 0.0147 0.0879 +/- 0.0102
CNeuroMod visual ROI all4    Gated ROI Transformer       0.6071 +/- 0.0423 0.0159 +/- 0.0034 0.0979 +/- 0.0184 0.0827 +/- 0.0111
```

BOLD5000 all-four-subject pair-level results, mean +/- std over 3 seeds:

```text
Pair       Model                       AUROC             AUPRC             R@5               MRR
CSI1-CSI2  ROI-MLP                     0.6652 +/- 0.0059 0.0082 +/- 0.0004 0.0442 +/- 0.0113 0.0466 +/- 0.0067
CSI1-CSI2  Gated ROI Transformer       0.6129 +/- 0.0548 0.0072 +/- 0.0012 0.0400 +/- 0.0090 0.0407 +/- 0.0029
CSI1-CSI3  ROI-MLP                     0.6755 +/- 0.0020 0.0103 +/- 0.0012 0.0708 +/- 0.0088 0.0637 +/- 0.0012
CSI1-CSI3  Gated ROI Transformer       0.6446 +/- 0.0618 0.0093 +/- 0.0024 0.0642 +/- 0.0188 0.0558 +/- 0.0132
CSI1-CSI4  ROI-MLP                     0.6995 +/- 0.0025 0.0100 +/- 0.0002 0.0517 +/- 0.0072 0.0578 +/- 0.0072
CSI1-CSI4  Gated ROI Transformer       0.7076 +/- 0.0058 0.0107 +/- 0.0002 0.0633 +/- 0.0118 0.0591 +/- 0.0029
CSI2-CSI3  ROI-MLP                     0.6110 +/- 0.0214 0.0092 +/- 0.0015 0.0467 +/- 0.0088 0.0483 +/- 0.0086
CSI2-CSI3  Gated ROI Transformer       0.5809 +/- 0.0377 0.0076 +/- 0.0010 0.0550 +/- 0.0066 0.0470 +/- 0.0088
CSI2-CSI4  ROI-MLP                     0.6477 +/- 0.0047 0.0084 +/- 0.0005 0.0567 +/- 0.0146 0.0526 +/- 0.0044
CSI2-CSI4  Gated ROI Transformer       0.6091 +/- 0.0697 0.0086 +/- 0.0031 0.0583 +/- 0.0123 0.0555 +/- 0.0090
CSI3-CSI4  ROI-MLP                     0.6377 +/- 0.0275 0.0103 +/- 0.0022 0.0533 +/- 0.0227 0.0508 +/- 0.0103
CSI3-CSI4  Gated ROI Transformer       0.5888 +/- 0.0479 0.0077 +/- 0.0020 0.0558 +/- 0.0170 0.0498 +/- 0.0146
```

CNeuroMod all-four-subject pair-level results, mean +/- std over 3 seeds:

```text
Pair           Model                       AUROC             AUPRC             R@5               MRR
sub-01-sub-02  ROI-MLP                     0.6148 +/- 0.0247 0.0149 +/- 0.0017 0.1044 +/- 0.0142 0.0825 +/- 0.0111
sub-01-sub-02  Gated ROI Transformer       0.5704 +/- 0.0396 0.0130 +/- 0.0022 0.0828 +/- 0.0100 0.0733 +/- 0.0088
sub-01-sub-03  ROI-MLP                     0.6350 +/- 0.0155 0.0164 +/- 0.0026 0.1089 +/- 0.0158 0.0894 +/- 0.0098
sub-01-sub-03  Gated ROI Transformer       0.6401 +/- 0.0137 0.0190 +/- 0.0041 0.1028 +/- 0.0067 0.0898 +/- 0.0046
sub-01-sub-06  ROI-MLP                     0.6136 +/- 0.0217 0.0176 +/- 0.0044 0.1017 +/- 0.0246 0.0847 +/- 0.0147
sub-01-sub-06  Gated ROI Transformer       0.5775 +/- 0.0408 0.0148 +/- 0.0036 0.0900 +/- 0.0233 0.0786 +/- 0.0158
sub-02-sub-03  ROI-MLP                     0.6357 +/- 0.0051 0.0163 +/- 0.0009 0.1156 +/- 0.0121 0.0939 +/- 0.0102
sub-02-sub-03  Gated ROI Transformer       0.6416 +/- 0.0083 0.0176 +/- 0.0014 0.1167 +/- 0.0174 0.0903 +/- 0.0089
sub-02-sub-06  ROI-MLP                     0.6034 +/- 0.0142 0.0147 +/- 0.0010 0.0944 +/- 0.0111 0.0846 +/- 0.0083
sub-02-sub-06  Gated ROI Transformer       0.5705 +/- 0.0273 0.0130 +/- 0.0013 0.0872 +/- 0.0167 0.0748 +/- 0.0097
sub-03-sub-06  ROI-MLP                     0.6460 +/- 0.0157 0.0184 +/- 0.0001 0.1100 +/- 0.0109 0.0924 +/- 0.0109
sub-03-sub-06  Gated ROI Transformer       0.6424 +/- 0.0192 0.0180 +/- 0.0026 0.1078 +/- 0.0169 0.0893 +/- 0.0068
```

Interpretation:

```text
The expanded public visual-ROI smoke checks confirm above-chance cross-subject same-image signal in CNeuroMod-THINGS and BOLD5000, but they do not replicate the main NSD conclusion that the gated ROI Transformer is consistently stronger than ROI-MLP. On CNeuroMod, ROI-MLP is stronger in the all-pair average, with the gated transformer competitive or slightly better on selected subject pairs. On BOLD5000, ROI-MLP is stronger on AUROC/AUPRC/MRR in the all-pair average, while R@5 is essentially mixed. These results are useful as external feasibility and stress-test evidence, and they strengthen the manuscript's limitation that full external validation should use trial-wise atlas/HCP-MMP beta maps rather than low-dimensional public visual-ROI summaries.
```

## THINGS-fMRI Progress On Shanghai HPC

THINGS-fMRI was selected as the next full external-validation candidate because it provides public single-trial fMRI response derivatives rather than only raw BIDS fMRI. The dataset is less directly matched to the current strict-repeat design than CNeuroMod-THINGS, but it is useful for a broader natural-object cross-subject retrieval stress test.

Public source status:

```text
Dataset: THINGS-fMRI1 / OpenNeuro ds004192
Design: 3 subjects, 12 sessions, approximately 8.7k natural object images from 720 object concepts
Useful derivative: THINGS-data fMRI single-trial responses, table format
Derivative DOI: 10.25452/figshare.plus.20492835.v2
Figshare API article: https://api.figshare.com/v2/articles/20492835
File ID: 43635873
File name: betas_csv.zip
File size: 42,955,087,212 bytes
Expected contents: sub-{subject}_ResponseData.h5, sub-{subject}_StimulusMetadata.csv, sub-{subject}_VoxelMetadata.csv
```

Metadata/code status:

```text
Code repository cloned on Shanghai HPC:
  /gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/things_fmri/metadata_repo

Repository remote:
  https://github.com/ViCCo-Group/THINGS-data.git

Useful code/notebooks:
  MRI/notebooks/fmri_usage.ipynb
  MRI/notebooks/working_with_rois.ipynb
  MRI/thingsmri/dataset.py
```

Download/conversion status:

```text
Target directory:
  /gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/things_fmri/figshare_table

Target archive:
  betas_csv.zip

Archive size:
  42,955,087,212 bytes

Exported scalar4 cache:
  /gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/things_fmri/visual_roi_scalar4_smoke

Subjects:
  sub-01, sub-02, sub-03

Common shared images:
  1000 per subject

Exported source visual ROIs:
  V1, V2, V3, hV4, VO1, VO2, TO1, TO2, lFFA, rFFA, lOFA, rOFA,
  lEBA, rEBA, lPPA, rPPA, lRSC, rRSC, lLOC, rLOC

QC:
  no NaN/Inf values in exported trial tensors
```

Access notes:

```text
The original plus.figshare.com ndownloader endpoint returned a WAF challenge from the HPC login node and failed DNS resolution from Slurm compute nodes.
The Figshare API successfully exposed the current file ID and download URL.
The working transfer was started directly on Shanghai HPC using:
  https://ndownloader.figshare.com/files/43635873
```

Training status:

```text
Completed Slurm training array:
  18/18 jobs completed successfully

Subject pairs:
  sub-01_sub-02, sub-01_sub-03, sub-02_sub-03

Models:
  ROI-MLP and gated ROI Transformer

Seeds:
  11, 22, 33

Summary files:
  /gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/summary/things_fmri_visual_roi_all_runs.csv
  /gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/summary/things_fmri_visual_roi_mean_std_by_pair_model.csv
  /gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/summary/things_fmri_visual_roi_mean_std_by_dataset_model.csv
  /gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/summary/things_fmri_visual_roi_summary.md
```

Prepared scripts:

```text
scripts/export_things_fmri_visual_roi_scalar4.py
scripts/shanghai_things_fmri_export_after_download.sbatch
scripts/shanghai_things_fmri_external_array.sbatch
scripts/submit_shanghai_things_fmri_external.sh
```

Overall THINGS-fMRI visual-ROI results, mean +/- std across subject-pair x seed runs:

```text
Dataset                   Model                       AUROC             AUPRC             R@5               MRR               n
THINGS-fMRI visual ROI    ROI-MLP                     0.5777 +/- 0.0352 0.0067 +/- 0.0011 0.0506 +/- 0.0159 0.0456 +/- 0.0092 9
THINGS-fMRI visual ROI    Gated ROI Transformer       0.5291 +/- 0.0377 0.0057 +/- 0.0008 0.0308 +/- 0.0132 0.0335 +/- 0.0076 9
```

THINGS-fMRI pair-level results, mean +/- std over 3 seeds:

```text
Pair           Model                       AUROC             AUPRC             R@5               MRR
sub-01_sub-02  ROI-MLP                     0.6137 +/- 0.0125 0.0077 +/- 0.0011 0.0650 +/- 0.0139 0.0533 +/- 0.0092
sub-01_sub-02  Gated ROI Transformer       0.5608 +/- 0.0408 0.0065 +/- 0.0006 0.0400 +/- 0.0050 0.0402 +/- 0.0027
sub-01_sub-03  ROI-MLP                     0.5546 +/- 0.0078 0.0060 +/- 0.0001 0.0358 +/- 0.0088 0.0381 +/- 0.0031
sub-01_sub-03  Gated ROI Transformer       0.5189 +/- 0.0298 0.0053 +/- 0.0005 0.0208 +/- 0.0118 0.0281 +/- 0.0056
sub-02_sub-03  ROI-MLP                     0.5646 +/- 0.0418 0.0064 +/- 0.0011 0.0508 +/- 0.0101 0.0455 +/- 0.0086
sub-02_sub-03  Gated ROI Transformer       0.5076 +/- 0.0281 0.0053 +/- 0.0006 0.0317 +/- 0.0159 0.0321 +/- 0.0088
```

Interpretation:

```text
THINGS-fMRI confirms that the public compact visual-ROI pipeline recovers above-chance cross-subject same-image signal on a third independent natural-image fMRI dataset. It does not support a universal gated ROI Transformer advantage in compact public visual-ROI summaries: ROI-MLP is stronger on all aggregate THINGS-fMRI metrics. Together with CNeuroMod and BOLD5000, this supports the manuscript's narrowed claim that the main gated ROI-token result is validated in the NSD HCP-MMP setting, while broader external validation requires trial-wise atlas/HCP-MMP beta maps or a stronger external ROI harmonization pipeline.
```

## NOD Progress On Shanghai HPC

NOD metadata and support code were cloned directly on Shanghai HPC:

```text
/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/nod
```

Useful properties:

```text
OpenNeuro dataset: ds004496
subjects: 30 total metadata subjects
image tasks: ImageNet and COCO
available support files: HCP-MMP visual-cortex CSV, MMP_mpmLR32k.mat, roilbl_mmp.csv, template dtseries
event metadata: available and parseable
```

Current blocker:

```text
The metadata repository does not include precomputed HCP-MMP beta tensors.
The NOD validation code derives beta maps from Ciftify/fMRIPrep time-series derivatives.
The OpenNeuro metadata clone contains fMRIPrep file pointers, but the current GitHub clone reports no annex copies for sampled large NIfTI files unless a proper OpenNeuro/DataLad remote is configured.
```

Interpretation:

```text
NOD is the best next candidate for a full HCP-MMP/fsLR external validation, but it requires a separate targeted download/preprocessing plan. It should not be treated as completed validation yet.
```

## Tool Requirements On Shanghai HPC

Current Shanghai environment has `nibabel` and `nilearn`, but does not have `datalad`, `git-annex`, or `h5py` by default. CNeuroMod-THINGS requires DataLad/git-annex for selective data access. BOLD5000/GOD-style HDF5 products require `h5py`.

Use:

```bash
cd /gpfsnyu/scratch/xh2906/ReGraph-VLM
source scripts/shanghai_env.sh
python -m pip install h5py datalad datalad-installer
```

`git-annex` may still need installation through an HPC-supported package manager or `datalad-installer`.

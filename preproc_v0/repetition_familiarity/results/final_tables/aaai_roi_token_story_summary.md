# AAAI ROI-Token Story Summary

Recommended claim: fixed anatomical ROI-token modeling, gated ROI-preserving readout, and image alignment improve cross-subject natural-image fMRI retrieval.

Do not claim that explicit fixed adjacency is the source of the gain. The no-adjacency gated ROI Transformer is statistically tied with the final BNT/ReGraph ROI-token variant, and static adjacency perturbations plus learned edge-bias follow-ups do not establish a separate fixed-edge contribution.

Core evidence:
- Gated ReGraph/BNT+CLIP and no-adj gated ROI Transformer+CLIP both outperform ROI-MLP+CLIP in the main all-fold setting.
- ROI-order shuffle and gate controls show that fixed ROI-token layout and learned gates are important.
- Single-reference session-matched controls preserve the ROI-token conclusion.
- External visual-ROI smoke checks show above-chance cross-subject signal outside NSD but are not full HCP-MMP external validations.

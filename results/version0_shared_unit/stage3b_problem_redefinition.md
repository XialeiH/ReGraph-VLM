# Stage 3B Problem Redefinition

## 1. Stage 3A Recap

**Objective.**  
Test whether explicit unit-unit interaction on top of learned shared units improves held-out-subject performance.

**Result.**

- Static interaction is trainable.
- Residual-gated versions are stable.
- But there is no consistent gain over `prototype main`.

**Conclusion.**

> Static unit interaction is not trivially beneficial in the averaged-response, closed-set identification setting.

## 2. Diagnosis: Why Stage 3A Did Not Yield Gains

### 2.1 Task mismatch

The current task uses:

- averaged repeated responses
- static representations
- closed-set classification

This setting primarily rewards:

- cross-subject alignment
- representation separability

It does not naturally force interaction modeling.

### 2.2 Representation already compressed

The current pipeline is:

`ROI -> PCA -> encoder -> prototype -> z (64-d)`

Interaction is applied after strong compression. That reduces feature redundancy and leaves limited room for an interaction layer to recover extra signal.

### 2.3 Interaction signal may be averaged out

The current setting uses:

- averaged beta responses
- no trial-level variability
- no temporal ordering

This weakens the kinds of fluctuations where interaction structure might matter:

- co-recruitment variability
- trial-to-trial variation
- response dynamics

### 2.4 Empirical signal summary

Stage 2 and Stage 3A together show:

- shared-unit modeling is useful
- cross-subject consistency exists
- interaction is trainable
- interaction does not yet provide stable gains

The most defensible reading is:

> In the current setting, interaction is an optional feature rather than a structurally required component.

## 3. Principle for Stage 3B

> Stage 3B must move to a setting where interaction is structurally necessary rather than optional.

In other words:

- Stage 3A asked whether interaction helps in the same averaged-response setting.
- Stage 3B asks whether there is a nearby setting where interaction becomes necessary.

## 4. Candidate Problem Settings

### 4.1 Setting A: trial-level (non-averaged) responses

**Change:**

- stop averaging repeated responses
- use single-trial or repeated-trial beta responses directly

**Why interaction may matter:**

- trial noise differs
- unit recruitment fluctuates across repeats
- the same stimulus across trials induces a structured variability pattern

Graph structure could then help with:

- denoising
- stabilizing co-recruitment
- modeling variability structure

### 4.2 Setting B: co-activation structure modeling

**Change:**

- move from pure classification to structure prediction or graph regularization

**Why interaction may matter:**

- unit coactivation already exists empirically
- graph structure could act as a dependency prior or auxiliary objective

### 4.3 Setting C: pseudo-temporal or coarse dynamics

**Change:**

- construct coarse temporal bins or short trial sequences

**Why interaction may matter:**

- unit recruitment may evolve across coarse temporal context
- graph structure could model state transitions rather than a static mapping

## 5. Stage 3B MVE Decision

The Stage 3B MVE will use:

> **Setting A: trial-level / non-averaged responses**

### Why this is the right MVE

1. Minimal change

- the raw NSD beta data already exist
- the current ROI feature pipeline can be reused
- labels do not need to be redefined

2. Interaction signal can naturally reappear

- trial variance is not only nuisance
- it may reveal unit recruitment variability and cross-trial structure

3. It is the most natural continuation of the current result

- Stage 3A showed that interaction is not clearly useful after averaging
- the next question is whether interaction becomes useful before averaging removes trial variability

## 6. Stage 3B MVE Spec

### Input

- trial-level beta responses
- no repeated-response averaging

### Model comparison

- `prototype main` as the baseline
- `prototype + light interaction` reusing the most stable Stage 3A graph core

### Graph

- static graph only
- no dynamic GNN in this stage

### Metrics

- top-1 accuracy
- top-5 accuracy
- optional cross-trial consistency diagnostics

## 7. Success and Failure Criteria

### Success

Any one of these is sufficient:

- interaction beats `prototype main` stably across folds
- or interaction improves stability in the noisier trial-level setting
- or interaction improves cross-subject consistency in the trial-level setting

### Failure

- interaction still provides no gain
- or training becomes unstable

If that happens, the conclusion becomes:

> interaction does not add value even when trial-level variability is present

## 8. What We Explicitly Do Not Do

- no dynamic GNN
- no multi-step temporal modeling
- no extension to THINGS-fMRI or BOLD5000
- no change to the prototype main structure itself
- no reopening of Stage 3A gate-centric tuning

## 9. Final Decision Rule

> If Stage 3B still does not show gains, the project will treat graph interaction as a secondary or analysis tool rather than a core modeling component for this problem.

## Final Statement

> Stage 3A established that interaction is feasible but not sufficient. Stage 3B therefore shifts the focus from improving the same setting to identifying a setting where interaction is intrinsically required.

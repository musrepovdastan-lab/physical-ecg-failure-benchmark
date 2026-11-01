# Amendment 001 — controlled initialization and replayable RNG

**Frozen before any corrected fit. Date: 2026-10-31.**

## Minimum implementation differences from calibration_role_control_v1

1. Set Python `random`, NumPy, PyTorch CPU and all CUDA seeds before `ECGResNetSmall()` is constructed. This corrects the confirmed initialization-order defect.
2. Enable deterministic PyTorch algorithms, deterministic cuDNN, disable cuDNN benchmark and TF32, and set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing PyTorch.
3. Give DataLoader shuffling an explicit per-fit generator; inference DataLoaders use a separate local generator. This prevents validation and prediction passes from consuming the training RNG stream.
4. Save a canonical named-tensor hash and initial `state_dict`, plus Python/NumPy/Torch CPU/CUDA RNG states before and after initialization and immediately before training.
5. Save and restore all RNG states and the DataLoader generator state in the atomic epoch-resume checkpoint.
6. Record actual training-only elapsed seconds and CUDA peak allocation separately from replay/inference time.
7. Validate RNG checkpoint/resume by comparing a fixed two-epoch miniature trajectory run continuously against one interrupted and resumed in a fresh process; model-state and final RNG-state digests must match.

The following scientific settings remain bound to the prior accepted protocol: data and row order, role memberships, train-only preprocessing, architecture, recipes and augmentation probabilities, loss/optimizer/batch size/AMP, checkpoint selection (TWA AUPRC on model_selection, max 30 epochs, patience 5, strict improvement, earliest tie), adequacy gates, endpoints, calibration and threshold rules, held-out conditions, bootstrap index files, metrics and contrasts.

## Input lineage notes

The prior protocol binds `NEXT_RESEARCH_DECISIONS_20261019.md`; the only local change since that freeze is its status sentence, with the full prior plan snapshotted in `ACTIVE_PLAN_BEFORE_POSITIONING.md`. The prior protocol also refers to root `LUNA_PROMPT_CALIBRATION_CONTROL_20261019.md`, which is not present in the current workspace. The current task arrived as an attached instruction and its hash is recorded in the corrected protocol. No scientific input file changed; the retained base protocol and accepted reports remain the source for fixed settings.

Initialization audit is required to pass before the nine fits. A matching initial-state hash is evidence about initialization only. This series does not claim that a complete 30-epoch trajectory is independently reproduced across processes; deterministic settings and complete resumable RNG snapshots are recorded instead.

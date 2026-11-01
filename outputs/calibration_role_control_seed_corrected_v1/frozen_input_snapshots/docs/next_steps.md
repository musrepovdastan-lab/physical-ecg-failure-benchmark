# Next step: correct initialization reproducibility

The identified technical follow-up is a corrected rerun of the calibration-role
control. In the original runner, the seed was set after model construction, so
it did not determine the initial weights.

1. Create a separate corrected runner: set Python, NumPy, Torch and CUDA seeds
   before constructing the model. Save the initial weights and RNG states.
2. Verify that the same seed gives identical initial weights across methods and
   repeated fresh processes, and that different seeds change the initialization.
3. Repeat the same nine fits: ordinary, physics-aware p050 and combined training,
   each with seeds 104729, 130363 and 155921. Keep the data roles, preprocessing,
   recipes and checkpoint-selection rules fixed.
4. Recompute predictions, calibration, thresholds and statistical comparisons
   from the new checkpoints, then report whether the conclusions change.

The original fits remain historical results. The corrected series will be a
reproducibility/sensitivity check on previously examined data.

The nine original fits took 70.3 GPU minutes in total on the local RTX 5070 Laptop
GPU. This is a reference for training cost; downstream evaluation takes additional
time. The corrected runner and rerun have not yet been completed.

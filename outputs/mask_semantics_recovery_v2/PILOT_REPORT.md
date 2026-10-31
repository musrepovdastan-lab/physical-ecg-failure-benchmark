# Ningbo mask recovery pilot

Decision: **PASS**

The frozen 512-record clean and RA-LL probe used `atol=1e-5`, `rtol=1e-4`. Ten of twelve saved-score checks passed within the unchanged tolerance. The two failed model/seed checks had their complete clean and RA-LL arrays re-inferred in the recovery cache.

Five-mask inference took 4.38 seconds (0.88 seconds per mask). The estimated 4,752-mask inference time was 1.16 hours. The exact weighted-bootstrap benchmark took 3.72 seconds for 20 resamples, 792 masks, and one endpoint; the initial full-statistics estimate was 0.93 hours.

Peak CUDA allocation was 92.9 MiB; the raw test matrix used 319.6 MiB. The baseline seed 104729 forensic inverted exact replay matched the historical cache exactly, for that single condition only.

No model training or data acquisition occurred.

# Amendment 005 — historical replay batch geometry

The first-32 replay initially fed a 32-row model batch. The frozen historical inference used batch 128, and the nonhistorical shape changed CUDA probability arithmetic enough to miss the 1e-4 replay criterion. The replay now runs the original 128-row batch and compares its first 32 rows, with the same checkpoint, autocast, normalization and model call path. Mean and last blocks retain their historical batch sizes. No full prediction file was written by the failed attempt.

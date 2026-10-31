# Amendment 002 — runner component binding

The launch guard stopped before the first epoch because the protocol binds each executable component separately, while the guard compared that mapping to a single file hash. The failure and exact frozen file are preserved in `FAILED_ATTEMPT_002.json` and `FROZEN_EXECUTION_PROTOCOL_FAILED_002.json`.

The guard now checks the hashes for `control.py` and `detector_scores.py` individually. Scientific settings, input memberships, and output scope are unchanged. The corrected protocol is frozen again before training.

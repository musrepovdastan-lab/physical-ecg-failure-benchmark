# Amendment 001 ? condition order

The first calculation attempt stopped before finalizing outputs because a saved-point comparison detected that the completion runner had ordered conditions differently from the frozen source. `control.py` defines the order as clean, RA?LA, RA?LL, LA?LL. The completion runner is corrected to this order. The protocol scope and saved inputs are unchanged. No training, inference, calibration fit, threshold selection, or v1 modification occurred. Details and hashes are in `FAILED_ATTEMPT_001.json` and `COMPLETION_PROTOCOL_AMENDMENT_001.json`.

# Engineering amendment 004

Huang detector inference must use the existing isolated TensorFlow environment, which does not install PyTorch. The first launch failed before inference because the shared runner imported PyTorch unconditionally. The runner now loads PyTorch only for diagnostic classifier commands. No dependency was installed and no model/protocol setting changed. Failed attempt preserved in `FAILED_ATTEMPT_004.json`.

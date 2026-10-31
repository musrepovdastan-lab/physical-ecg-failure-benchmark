# Evidence audit

| Evidence | Chapman M1 | Ningbo M2 |
|---|---:|---:|
| Decision | PASS_M1_FREEZE_NINGBO_M2 | PASS_M2_EXTERNAL_REPLICATION |
| Protocol hash verified | 248976F691003DAA45C5F188C53F7CF75891F19F8723223477FC41E1D65667CE | 2D7B605AF175EF6D5B59B223A9EA37BB3F7D675C2C1AC3E27FFF23FCD43E96B8 |
| Seeds | 3 | 3 |
| Validation gates | PASS | PASS |
| Conditions per seed | 794 | 794 |
| Bootstrap replicates | 1000 | 1000 |
| Premature test access | none | none |

M2 verification rehashed and shape/finite-checked 2382 saved prediction files. The M2 bootstrap state hash is `8DE562E38F9AB637640D491EFAEF67B323B2DFBA54FCC210FEE5C37C8092C891` and is bound to the reconciled COMPLETE telemetry. This audit read only cached artifacts; it did not load raw ECG data, train a model, invoke model inference, regenerate predictions, or recompute bootstrap replicates.

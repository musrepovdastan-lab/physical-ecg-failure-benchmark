# ECGFounder label mapping audit — v1

The pinned official `tasks.txt` has 150 lines. The official evaluation script reads this
file in order and uses the same 150-output model. The checkpoint has a 150-by-1024
`dense.weight`, and its state dictionary loads without missing or unexpected keys
into the authors' 12-lead `Net1D` configuration.

| Benchmark endpoint | Existing benchmark label | Official output | Decision |
|---|---|---|---|
| TWA (primary) | SNOMED 164934002, T-wave abnormality | `NONSPECIFIC T WAVE ABNORMALITY` at zero-based index 14; several other T-wave change statements | **No exact output.** Do not score any of these as TWA zero-shot. Use the dataset-trained frozen-backbone head for the exact-label comparison. |
| Sinus bradycardia (SB) | SNOMED 426177001 | `SINUS BRADYCARDIA`, index 4; `MARKED SINUS BRADYCARDIA` is a separate class | Name-level match. Report direct output separately; main comparison uses the exact-label trained head. |
| Sinus tachycardia (STach) | SNOMED 427084000 | `SINUS TACHYCARDIA`, index 6 | Name-level match. Report direct output separately; main comparison uses the exact-label trained head. |

The official labels are free-text diagnostic statements rather than the benchmark's
SNOMED-coded annotations. A name-level match does not establish identical annotation
rules or prevalence. The pretrained zero-shot results for SB and STach are therefore
secondary and separate from the exact-label head results.

Source: authors' pinned [tasks.txt](official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/tasks.txt),
[ptbxl_eval.py](official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/ptbxl_eval.py),
and [checkpoint](https://huggingface.co/PKUDigitalHealth/ECGFounder/blob/main/12_lead_ECGFounder.pth).

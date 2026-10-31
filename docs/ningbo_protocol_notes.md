# Ningbo confirmation protocol v2 amendment

V1 remains frozen and stopped at `STOP_DUPLICATE_CONFLICT`. Before any v1 split,
model training, or test evaluation, its audit found exactly three conflicting
waveform-hash groups containing six source records. V2 is frozen after that
pre-split data-quality audit and binds the v1 protocol and decision.

The sole substantive change is deterministic exclusion, before splitting, of
every member of each exact waveform-hash group whose age, sex, or complete Dx
signature conflicts. The rule is independent of TWA, SB, and STach labels.
It requires exactly three groups and six excluded records, otherwise stops for
data integrity. Target-label concordance is recorded only and cannot affect
membership. All other source, endpoint, split, model, gate, fault, comparator,
bootstrap, and decision specifications are unchanged.

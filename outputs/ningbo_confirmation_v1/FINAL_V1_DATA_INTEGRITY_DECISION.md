# STOP_DUPLICATE_CONFLICT

The frozen v1 ingestion audit parsed 34,905 source records and found 34,902
unique unmodified-waveform SHA-256 values. Three duplicate waveform groups
(six records) have conflicting age, sex, or complete diagnosis-code metadata.
There were zero exact-metadata duplicate groups and zero groups with a conflict
in the prespecified target endpoint labels. Under frozen v1 this requires
`STOP_DUPLICATE_CONFLICT`.

Conflict artifact SHA-256: `F5976880484276C1747B1A3598EE92322733273BE92DCA5AA04EC7C2A0E0CF2F`.
Frozen v1 protocol SHA-256: `B6EEB79C0CD214FE7F41FDC526AEC64DF2E9D982F59A45E7F21D0108B5A3698E`.

No split, training, or test evaluation occurred under v1.

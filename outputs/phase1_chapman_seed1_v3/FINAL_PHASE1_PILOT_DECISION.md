# Phase 1 Chapman-only pilot v3 — final decision

## STOP_DATA_INTEGRITY

The official Chapman-Shaoxing acquisition is complete and checksum-valid:
10,247 records, 10,247 headers, 10,247 waveform files, 1,237,539,902 verified
bytes, zero checksum mismatches, and zero `.part` files. No Ningbo file was
accessed.

The strict audit found 13 exact waveform hashes shared by different record IDs.
There are no malformed headers, non-finite waveforms, missing age/sex values, or
duplicate record IDs. However, the frozen v3 protocol permits neither silent
deduplication nor undocumented exclusions. This is therefore an integrity-stop,
not a negative model result.

No split was created, no model was trained, and no Chapman internal-test outcome
was accessed. The only permitted next step is a protocol amendment and re-freeze
that explicitly specifies whether and how duplicate waveforms may be handled.

# Data sources and local layout

No raw ECGs or individual-level tables are distributed here.

| Source | Access and local location |
| --- | --- |
| Chapman-Shaoxing | Official [PhysioNet Challenge 2021 v1.0.3](https://physionet.org/content/challenge-2021/1.0.3/); `data/challenge2021_v103/chapman_shaoxing/` |
| Ningbo | Same official release; `data/challenge2021_v103/ningbo/` |
| HEEDB I0006 | Separately authorized restricted access; `data/heedb_restricted/I0006/` |

Follow each source's access and license terms. Dataset redistribution rights are
separate from access to this code. HEEDB metadata, record selections, labels,
waveforms, credentials and predictions are excluded.

The root `DATA_MANIFEST.json` describes the historical Phase-0 Challenge 2020 audit,
not later downloads. Later studies have their own protocols and ingestion audits.

Research runners expect local record maps, duplicate handling, split memberships,
normalization, checkpoints and predictions at the original study paths. Recreate
these prerequisites using the relevant protocol and runner. Full experiments need
downloads, compute and, for HEEDB, restricted access.

# Acquisition status — controlled pause

The Stage 2 acquisition gate passed before transfer: the official Challenge 2021
v1.0.3 release's eleven public Chapman-Shaoxing group indices enumerate exactly
10,247 records, and its `SHA256SUMS.txt` contains the matching record files.
The allow-list is restricted in code to `training/chapman_shaoxing/g1` through
`g11`; no Ningbo URL is constructed.

The execution host imposes a 30-second command-supervision ceiling. Direct
per-file requests to the official release exceed that ceiling and the host
detaches the running process. Because this pilot explicitly forbids background
jobs, acquisition was stopped rather than allowed to continue unobserved.

At the controlled stop, `g1` contained 103 `.hea` and 103 `.mat` files. Each
completed file was verified against the official v1.0.3 SHA-256 manifest before
being committed. There are no `.part` files. This is deliberately an incomplete
source and **is not eligible for ingestion, splitting, training, or test access**.

The resumable foreground command for a normal interactive terminal is:

```powershell
& .\.venv\Scripts\python.exe .\src\chapman_stage23.py download-group 1 --workers 8
```

Run it for groups 1 through 11, one foreground command at a time. The utility
rechecks existing file hashes and fails on any mismatch; once all groups finish,
run `audit` and then `split`. No data repair, exclusion, model prediction, or
Ningbo access has occurred.

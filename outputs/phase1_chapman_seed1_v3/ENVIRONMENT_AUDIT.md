# Stage 1 environment audit — Chapman-only pilot v3

Timestamp (local): 2026-08-28

## Frozen protocol

- File: `configs/FROZEN_PHASE1_PILOT_PROTOCOL_v3.json`
- SHA-256: `289838FC500DDD118D664A9F3909D71476FE09CF7A43C597AD198313C0489C09`
- Parsed JSON version: `phase1_chapman_seed1_v3`
- This audit was completed before any Chapman model prediction or waveform download.

## Project-local runtime

- Python: 3.12.10 (`.venv` in this project)
- PyTorch: `2.11.0+cu128`
- CUDA available: `True`
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU
- CUDA float16 autocast smoke test: passed (`torch.float16` output)
- Physics unit tests: `4 passed in 0.32s`
- C: free space at gate: 185,136,201,728 bytes (about 185 GB)

## Gate decision

**PASS — Stage 2 may begin.** The installation is project-local. CUDA, mixed precision,
physics tests, frozen-config integrity, and disk-space gates all passed. No Ningbo
waveforms were downloaded or accessed.

# v7 environment and integrity audit

- Frozen protocol SHA-256: `312DABB9A0400675097D492F66DF58C9DA5EAD69623FFF0563F4DAECE0944CB7`.
- Existing canonical split, preprocessing, physics library, seed-104729 checkpoint,
  and v6 cached predictions were retained and hashed in `FROZEN_PROTOCOL_HASH.json`.
- CUDA: available on NVIDIA GeForce RTX 5070 Laptop GPU.
- No duplicate v6 evaluator/trainer was active before v7 preparation.
- Ningbo was not accessed. Seed 104729 is reused; only 130363 and 155921 are
  authorized for training after this freeze.

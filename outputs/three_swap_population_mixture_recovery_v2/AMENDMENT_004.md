# Amendment 004 — batch normalization wrapper

The frozen per-record diagnostic input adapter does not accept a batch dimension. The recovery runner now performs the identical stride-5 and per-channel float32 normalization vectorized over the historical batch, and checks its first row against the frozen adapter. This changes only batching syntax, not preprocessing values, stride, dtype, or model path. No predictions were written by the failed attempt.

# Engineering amendment 002

The replay checker used waveform records from a separate Ningbo test order. The source multicondition score cache binds to `outputs/ningbo_confirmation_v2` and its own frozen membership. Replay now loads records through the exact frozen Ningbo manifest and the source waveform reader (`src.ningbo_confirmation_v1`), including the source model input sampling path. The 1e-4 tolerance and frozen model/cache bindings are unchanged. The failed attempt is preserved in `FAILED_ATTEMPT_002.json`; no scientific protocol parameter changed.

# Amendment 008 — safe identifier metadata

The six probability arrays had completed inference and replay, but their `file_names` metadata used NumPy object dtype. Convert identifiers to Unicode and repack the NPZ files; check the probability-array hash before and after. This does not alter model scores, labels, record order or scientific results.

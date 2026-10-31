# Amendment 012 — match saved inference batch geometry in replay

The first replay used 32-row probe batches, while full predictions were produced in 128-row batches. The Huang handoff itself matched the independent formula exactly, but half-precision model outputs differed by up to `2.44e-4` under the changed batch shape. The original prediction files and 1e-4 replay tolerance are preserved.

The replay now evaluates complete source-aligned 128-row first and middle batches plus the original final batch (which may be shorter). It compares stored outputs for the corresponding source records without changing precision, weights, or tolerance.

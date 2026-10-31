# FINAL INTEGRATED DEPLOYMENT DECISION

## Decision

Yes, the operational latency advantage is confirmed for this frozen implementation on this CPU. The warm integrated physics-aware worker had median/p95 latency 10.59/12.42 ms, versus 31.51/34.16 ms for the complete Huang detector, predicted correction, and clean-ResNet diagnosis request (2.97x median ratio). These are direct request-to-diagnostic-response measurements; no component times were summed.

The largest sampled process-tree working set was 612.2 MiB for physics-aware and 953.6 MiB for Huang (36% lower for the single model). Loading to ready was 5.82 s versus 10.98 s.

## Compact table

| Measure | Physics-aware single model | Huang detect-correct-diagnose |
|---|---:|---:|
| Warm batch-1 median | 10.59 ms | 31.51 ms |
| Warm batch-1 p95 | 12.42 ms | 34.16 ms |
| Run-median mean +/- SD | 10.59 +/- 0.08 ms | 31.47 +/- 0.17 ms |
| Run-median range | 10.51-10.76 ms | 31.21-31.69 ms |
| Run-p95 mean +/- SD | 13.16 +/- 3.10 ms | 34.04 +/- 0.64 ms |
| Run-p95 range | 11.71-20.15 ms | 33.05-34.77 ms |
| Loading to ready | 5.82 s | 10.98 s |
| Maximum sampled process-tree working set | 612.2 MiB | 953.6 MiB |
| Model artifacts | 3.70 MiB | 3.81 MiB |
| External pipe echo median | 0.056 ms | 0.054 ms |

## Output agreement

Before timing, all 128 balanced fixture ECGs were compared with the frozen v2 seed-104729 caches. Physics-aware threshold decisions agreed 100.00% with maximum probability difference 0.0004. Huang threshold decisions agreed 100.00%, detector classes agreed 100.00%, and maximum diagnostic probability difference was 0.0003.

## How to describe this to the mentor

Say: 'On our frozen Ningbo implementation, the physics-aware single model and the published Huang detector chained to our clean ResNet had statistically unresolved TWA AUPRC, but the single model was operationally simpler and 2.97x faster in a direct warm CPU request-to-response benchmark. This is a comparison with our integration of Huang's published detector weights, not a comparison with latency or diagnostic scores reported in the Huang paper.'

Do not describe the methods as equivalent and do not claim SOTA or clinical superiority.

## Missing evidence

A stronger conference submission still needs real rather than simulated cable reversals, independent multi-device and multi-center validation, a prospectively specified noninferiority/equivalence hypothesis for diagnostic performance, deployment measurements on the intended clinical hardware and software stack, and workflow evidence at measured clinical prevalence. The current IPC design is reproducible and fair for these frozen Python environments, but it is not an optimized production implementation of either method.

## Reproduce

From the project root on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe src\integrated_deployment_v4.py
```

Hardware: AMD Ryzen AI 7 350 w/ Radeon 860M, 16 logical processors, Windows-11-10.0.26200-SP0. Both frameworks were restricted to one thread and logical CPU 0; seven runs of 128 balanced ECGs were timed after output verification and warmup.

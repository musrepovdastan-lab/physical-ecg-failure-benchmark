# Amendment 004 — preserve Huang's 500-Hz incoming signal

The first postprocessor attempt correctly stopped because it supplied the 100-Hz Ningbo classifier cache to an adapter that requires original 500-Hz ECGs. No Huang score, calibration fit, test, or HEEDB result was produced by that attempt. Core A/B clean scores used the correct 100-Hz input and remain bound to locked models; they are safely reproducible.

The revised postprocessor loads canonical 500-Hz source ECGs for the Huang detector/correction chain and uses the adapter's fixed stride-5 normalization before the classifier. It does not upsample, and preserves the damaged incoming waveform for the predicted inverse step. Scope and memberships remain unchanged.
